//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: cache of linked shader program binaries, in RAM (LRU with a byte budget) and on disk (written on a worker thread).
// See the header for details.
//------------------------------------------------------------------------------------------------------------------------------------------
#include "GShaderCache.h"

#if PSYDOOM_GLES_RENDERER

#include "Asserts.h"
#include "PsyDoom/Utils.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <list>
#include <md5.h>
#include <mutex>
#include <SDL_log.h>
#include <thread>
#include <unordered_map>

BEGIN_NAMESPACE(GShaderCache)

// Cache file layout: a header followed by the program binary.
// Bump the version whenever the format of the files or anything else that goes into the keys changes.
static constexpr uint32_t   FILE_MAGIC          = 0x43534450;       // 'PDSC'
static constexpr uint32_t   FILE_VERSION        = 1;
static constexpr const char FILE_EXT[]          = ".bin";
static constexpr const char TMP_FILE_EXT[]      = ".tmp";
static constexpr const char CACHE_SUBFOLDER[]   = "gles_shader_cache";
static constexpr size_t     DEFAULT_RAM_BUDGET  = 8 * 1024 * 1024;  // Plenty for all of the programs of the renderer (a few MiB)

struct FileHeader {
    uint32_t    magic;
    uint32_t    version;
    uint32_t    binaryFormat;
    uint32_t    binarySize;
    uint64_t    payloadChecksum;    // FNV-1a 64 of the binary
    uint8_t     driverHash[16];     // MD5 of the driver identity: which driver made the binary
};

static_assert(sizeof(FileHeader) == 40);

// RAM cache: least recently used entries are at the back of the list
struct RamEntry {
    std::string     key;
    ProgramBinary   binary;
};

static std::list<RamEntry>                                          gRamLru;
static std::unordered_map<std::string, std::list<RamEntry>::iterator> gRamIndex;
static size_t                                                       gRamBytes;
static std::atomic<size_t>                                          gRamBudget = DEFAULT_RAM_BUDGET;     // Can be set from another thread

// Disk cache: the folder (with a trailing separator, empty if no disk cache) and the driver the binaries are for
static std::mutex       gCacheFolderMutex;
static std::string      gCacheFolderOverride;   // Set by the launcher (Android: through JNA)
static std::string      gCacheDir;
static std::string      gDriverId;
static uint8_t          gDriverHash[16];
static bool             gbInitialized;

// Disk jobs done by the worker thread
struct DiskJob {
    enum class Type : uint8_t {
        Write,          // Write the binary of a program
        Delete,         // Delete the file of a program
        Cleanup         // Delete leftover temporary files and the files made by other drivers or older versions
    };

    Type            type;
    std::string     key;
    ProgramBinary   binary;
};

static std::mutex               gJobsMutex;
static std::condition_variable  gJobsCondVar;
static std::deque<DiskJob>      gJobs;
static bool                     gbQuitWorker;
static std::thread              gWorkerThread;

//------------------------------------------------------------------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------------------------------------------------------------------
static uint64_t fnv1a64(const uint8_t* const pData, const size_t size) noexcept {
    uint64_t hash = 0xCBF29CE484222325ull;

    for (size_t i = 0; i < size; ++i) {
        hash ^= pData[i];
        hash *= 0x100000001B3ull;
    }

    return hash;
}

static std::string getFilePath(const std::string& key, const char* const ext) noexcept {
    return gCacheDir + key + ext;
}

// Checks a header read from a cache file against the current version and driver
static bool isHeaderForThisDriver(const FileHeader& header) noexcept {
    return (
        (header.magic == FILE_MAGIC) &&
        (header.version == FILE_VERSION) &&
        (std::memcmp(header.driverHash, gDriverHash, sizeof(gDriverHash)) == 0)
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Worker thread: disk jobs
//------------------------------------------------------------------------------------------------------------------------------------------
static void doWriteJob(const DiskJob& job) noexcept {
    ASSERT(job.binary.pData);
    const std::vector<uint8_t>& data = *job.binary.pData;

    FileHeader header = {};
    header.magic = FILE_MAGIC;
    header.version = FILE_VERSION;
    header.binaryFormat = job.binary.format;
    header.binarySize = (uint32_t) data.size();
    header.payloadChecksum = fnv1a64(data.data(), data.size());
    std::memcpy(header.driverHash, gDriverHash, sizeof(gDriverHash));

    // Write to a temporary file first and rename it once complete: a file with the final name is always a complete one
    const std::string tmpPath = getFilePath(job.key, TMP_FILE_EXT);
    const std::string finalPath = getFilePath(job.key, FILE_EXT);

    FILE* const pFile = std::fopen(tmpPath.c_str(), "wb");

    if (!pFile)
        return;

    const bool bWroteOk = (
        (std::fwrite(&header, sizeof(header), 1, pFile) == 1) &&
        (data.empty() || (std::fwrite(data.data(), data.size(), 1, pFile) == 1)) &&
        (std::fflush(pFile) == 0)
    );

    const bool bClosedOk = (std::fclose(pFile) == 0);
    std::error_code errorCode;

    if (bWroteOk && bClosedOk) {
        std::filesystem::rename(tmpPath, finalPath, errorCode);

        if (!errorCode)
            return;
    }

    std::filesystem::remove(tmpPath, errorCode);
}

static void doDeleteJob(const DiskJob& job) noexcept {
    std::error_code errorCode;
    std::filesystem::remove(getFilePath(job.key, FILE_EXT), errorCode);
}

static void doCleanupJob() noexcept {
    std::error_code errorCode;
    std::filesystem::directory_iterator dirIter(gCacheDir, errorCode);

    if (errorCode)
        return;

    for (const std::filesystem::directory_entry& entry : dirIter) {
        if (!entry.is_regular_file(errorCode))
            continue;

        const std::filesystem::path& path = entry.path();
        const std::string ext = path.extension().string();

        if (ext == TMP_FILE_EXT) {
            // Leftover of a write that never finished (app killed)
            std::filesystem::remove(path, errorCode);
        }
        else if (ext == FILE_EXT) {
            // Delete the binaries of other drivers or older cache versions: they can never be used again
            FileHeader header = {};
            FILE* const pFile = std::fopen(path.string().c_str(), "rb");
            bool bKeepFile = false;

            if (pFile) {
                bKeepFile = ((std::fread(&header, sizeof(header), 1, pFile) == 1) && isHeaderForThisDriver(header));
                std::fclose(pFile);
            }

            if (!bKeepFile) {
                std::filesystem::remove(path, errorCode);
            }
        }
    }
}

static void workerThreadMain() noexcept {
    while (true) {
        DiskJob job;

        {
            std::unique_lock<std::mutex> lock(gJobsMutex);
            gJobsCondVar.wait(lock, []() noexcept { return (gbQuitWorker || (!gJobs.empty())); });

            // Note: queued jobs are always finished before quitting, so all binaries made in this run are saved
            if (gJobs.empty())
                return;

            job = std::move(gJobs.front());
            gJobs.pop_front();
        }

        switch (job.type) {
            case DiskJob::Type::Write:      doWriteJob(job);    break;
            case DiskJob::Type::Delete:     doDeleteJob(job);   break;
            case DiskJob::Type::Cleanup:    doCleanupJob();     break;
        }
    }
}

static void queueDiskJob(DiskJob&& job) noexcept {
    if (gCacheDir.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(gJobsMutex);
        gJobs.push_back(std::move(job));
    }

    gJobsCondVar.notify_one();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// RAM cache
//------------------------------------------------------------------------------------------------------------------------------------------
static size_t getBinarySize(const ProgramBinary& binary) noexcept {
    return (binary.pData) ? binary.pData->size() : 0;
}

static void ramRemove(const std::string& key) noexcept {
    const auto indexIter = gRamIndex.find(key);

    if (indexIter == gRamIndex.end())
        return;

    gRamBytes -= getBinarySize(indexIter->second->binary);
    gRamLru.erase(indexIter->second);
    gRamIndex.erase(indexIter);
}

// Evicts the least recently used binaries until within the budget
static void ramEvictToBudget() noexcept {
    const size_t ramBudget = gRamBudget.load();

    while ((gRamBytes > ramBudget) && (!gRamLru.empty())) {
        RamEntry& entry = gRamLru.back();
        gRamBytes -= getBinarySize(entry.binary);
        gRamIndex.erase(entry.key);
        gRamLru.pop_back();
    }
}

static void ramAdd(const std::string& key, const ProgramBinary& binary) noexcept {
    ramRemove(key);

    // Don't keep anything that would not fit in the budget on its own
    const size_t binarySize = getBinarySize(binary);

    if (binarySize > gRamBudget.load())
        return;

    gRamLru.push_front(RamEntry{ key, binary });
    gRamIndex[key] = gRamLru.begin();
    gRamBytes += binarySize;
    ramEvictToBudget();
}

static bool ramFind(const std::string& key, ProgramBinary& binaryOut) noexcept {
    const auto indexIter = gRamIndex.find(key);

    if (indexIter == gRamIndex.end())
        return false;

    // Now the most recently used one
    gRamLru.splice(gRamLru.begin(), gRamLru, indexIter->second);
    binaryOut = gRamLru.front().binary;
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Disk cache: reads a binary (synchronously, it is needed right away) and validates it
//------------------------------------------------------------------------------------------------------------------------------------------
static bool diskFind(const std::string& key, ProgramBinary& binaryOut) noexcept {
    if (gCacheDir.empty())
        return false;

    const std::string path = getFilePath(key, FILE_EXT);
    FILE* const pFile = std::fopen(path.c_str(), "rb");

    if (!pFile)
        return false;

    FileHeader header = {};
    auto pData = std::make_shared<std::vector<uint8_t>>();
    bool bValid = ((std::fread(&header, sizeof(header), 1, pFile) == 1) && isHeaderForThisDriver(header) && (header.binarySize > 0));

    if (bValid) {
        pData->resize(header.binarySize);
        bValid = (
            (std::fread(pData->data(), header.binarySize, 1, pFile) == 1) &&
            (std::fgetc(pFile) == EOF) &&                                   // Nothing after the binary either
            (fnv1a64(pData->data(), pData->size()) == header.payloadChecksum)
        );
    }

    std::fclose(pFile);

    if (!bValid) {
        // Corrupt, truncated or not ours: delete it, it gets written again once the program is compiled
        queueDiskJob(DiskJob{ DiskJob::Type::Delete, key, {} });
        return false;
    }

    binaryOut.format = header.binaryFormat;
    binaryOut.pData = std::move(pData);
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the folder for the disk cache, a subfolder is made in it: on Android the app's cache folder (set by the launcher through JNA).
// Must be called before the renderer starts to take effect, otherwise a subfolder of the user data folder is used.
//------------------------------------------------------------------------------------------------------------------------------------------
void setCacheFolder(const char* const path) noexcept {
    std::lock_guard<std::mutex> lock(gCacheFolderMutex);
    gCacheFolderOverride = (path) ? path : "";
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the most memory the RAM cache may use for binaries (can be called from any thread).
// If the cache is over the new budget, the least recently used binaries are evicted when the next one is added.
//------------------------------------------------------------------------------------------------------------------------------------------
void setRamCacheBudget(const size_t numBytes) noexcept {
    gRamBudget = numBytes;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Starts the cache for the given driver (its vendor, renderer and version strings) and starts the disk worker thread.
// The disk cache is not used if not allowed (the driver can't give program binaries) or if the folder can't be made.
//------------------------------------------------------------------------------------------------------------------------------------------
void init(const std::string& driverId, const bool bDiskCacheAllowed) noexcept {
    if (gbInitialized) {
        // Already running (renderer restarted): keep the RAM cache, unless the binaries are now for another driver
        if (driverId == gDriverId)
            return;

        shutdown();
    }

    gbInitialized = true;
    gDriverId = driverId;

    {
        MD5 md5;
        md5.add(driverId.data(), driverId.size());
        md5.getHash(gDriverHash);
    }

    // Decide the folder for the disk cache and make it
    gCacheDir.clear();

    if (bDiskCacheAllowed) {
        std::string baseFolder;

        {
            std::lock_guard<std::mutex> lock(gCacheFolderMutex);
            baseFolder = gCacheFolderOverride;
        }

        if (baseFolder.empty()) {
            baseFolder = Utils::getOrCreateUserDataFolder();
        }

        if ((!baseFolder.empty()) && (baseFolder.back() != '/') && (baseFolder.back() != '\\')) {
            baseFolder += '/';
        }

        const std::string cacheDir = baseFolder + CACHE_SUBFOLDER + "/";
        std::error_code errorCode;
        std::filesystem::create_directories(cacheDir, errorCode);

        if (std::filesystem::is_directory(cacheDir, errorCode)) {
            gCacheDir = cacheDir;
        } else {
            SDL_Log("PsyDoom GLES: can't make the shader cache folder '%s', the shader cache will only be kept in RAM", cacheDir.c_str());
        }
    }

    // Start the worker thread and get it to tidy up the cache folder first
    if (!gCacheDir.empty()) {
        gbQuitWorker = false;
        gWorkerThread = std::thread(workerThreadMain);
        queueDiskJob(DiskJob{ DiskJob::Type::Cleanup, {}, {} });
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Finishes the disk writes still queued, stops the worker thread and frees the RAM cache
//------------------------------------------------------------------------------------------------------------------------------------
void shutdown() noexcept {
    if (gWorkerThread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(gJobsMutex);
            gbQuitWorker = true;
        }

        gJobsCondVar.notify_one();
        gWorkerThread.join();
    }

    gJobs.clear();
    gbQuitWorker = false;
    gRamLru.clear();
    gRamIndex.clear();
    gRamBytes = 0;
    gCacheDir.clear();
    gDriverId.clear();
    std::memset(gDriverHash, 0, sizeof(gDriverHash));
    gbInitialized = false;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Makes the key of a program from its shader sources (in order) and the driver identity
//------------------------------------------------------------------------------------------------------------------------------------------
std::string makeKey(const char* const* const pSources, const uint32_t numSources) noexcept {
    MD5 md5;
    md5.add(&FILE_VERSION, sizeof(FILE_VERSION));
    md5.add(gDriverId.data(), gDriverId.size());

    for (uint32_t i = 0; i < numSources; ++i) {
        const char* const pSource = pSources[i];
        const uint32_t length = (uint32_t) std::strlen(pSource);
        md5.add(&length, sizeof(length));   // So that the boundaries between the sources count too
        md5.add(pSource, length);
    }

    return md5.getHash();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Looks for the binary of a program: in RAM first, then on disk (a binary found there is kept in RAM too)
//------------------------------------------------------------------------------------------------------------------------------------------
bool find(const std::string& key, ProgramBinary& binaryOut) noexcept {
    ASSERT(gbInitialized);

    if (ramFind(key, binaryOut))
        return true;

    if (diskFind(key, binaryOut)) {
        ramAdd(key, binaryOut);
        return true;
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Adds the binary of a newly linked program: kept in RAM right away and written to disk in the background
//------------------------------------------------------------------------------------------------------------------------------------------
void add(const std::string& key, const ProgramBinary& binary) noexcept {
    ASSERT(gbInitialized);

    if ((!binary.pData) || binary.pData->empty())
        return;

    ramAdd(key, binary);
    queueDiskJob(DiskJob{ DiskJob::Type::Write, key, binary });
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Forgets the binary of a program (the driver rejected it), in RAM and on disk
//------------------------------------------------------------------------------------------------------------------------------------------
void remove(const std::string& key) noexcept {
    ASSERT(gbInitialized);
    ramRemove(key);
    queueDiskJob(DiskJob{ DiskJob::Type::Delete, key, {} });
}

END_NAMESPACE(GShaderCache)

//------------------------------------------------------------------------------------------------------------------------------------------
// Android: the launcher sets the folder for the shader cache (its cache folder) and the RAM cache budget through JNA
//------------------------------------------------------------------------------------------------------------------------------------------
#if ANDROID
extern "C" {
__attribute__((used)) __attribute__((visibility("default")))
void setPathToCacheFolder(const char* pathToCacheFolder) {
    GShaderCache::setCacheFolder(pathToCacheFolder);
}

__attribute__((used)) __attribute__((visibility("default")))
void setShaderRamCacheSizeKb(const int sizeKb) {
    GShaderCache::setRamCacheBudget((size_t) ((sizeKb > 0) ? sizeKb : 0) * 1024u);
}
}
#endif

#endif  // #if PSYDOOM_GLES_RENDERER
