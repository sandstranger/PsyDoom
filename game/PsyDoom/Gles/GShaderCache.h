#pragma once

#if PSYDOOM_GLES_RENDERER

#include "Macros.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

//------------------------------------------------------------------------------------------------------------------------------------------
// Cache of linked shader program binaries for the GLES renderer (glGetProgramBinary/glProgramBinary), so that the shaders are compiled
// only once per driver and not on every startup. Two levels:
//
//  (1) RAM: binaries kept in memory, least recently used first out when over a byte budget ('setRamCacheBudget').
//      Used when the renderer recreates programs while running (renderer restart, GL context lost and recreated by SDL).
//  (2) Disk: one file per program in the cache folder. On Android that folder is set by the launcher through JNA ('setPathToCacheFolder',
//      normally 'Context.getCacheDir()'), elsewhere it is in the user data folder. Files are written asynchronously on a worker thread
//      (written to a temporary file and renamed, so a killed app never leaves a half written binary), and are validated on load
//      (format version, driver identity, size, checksum), since a bad binary can crash some drivers.
//
// The key of a program covers its shader sources and the driver identity (vendor, renderer, version strings): a driver update gives
// new keys, and the files of other drivers are deleted by the worker thread on startup.
// This module makes no GL calls: binaries are fetched and given to the driver by 'GShaders'. All functions other than
// 'setCacheFolder' and 'setRamCacheBudget' must be called from the thread the renderer runs on.
//------------------------------------------------------------------------------------------------------------------------------------------
BEGIN_NAMESPACE(GShaderCache)

// A linked program binary: the driver specific format and the bytes (shared, so cache levels and queued writes hold no copies)
struct ProgramBinary {
    uint32_t                                        format;
    std::shared_ptr<const std::vector<uint8_t>>     pData;
};

void setCacheFolder(const char* const path) noexcept;
void setRamCacheBudget(const size_t numBytes) noexcept;
void init(const std::string& driverId, const bool bDiskCacheAllowed) noexcept;
void shutdown() noexcept;
std::string makeKey(const char* const* const pSources, const uint32_t numSources) noexcept;
bool find(const std::string& key, ProgramBinary& binaryOut) noexcept;
void add(const std::string& key, const ProgramBinary& binary) noexcept;
void remove(const std::string& key) noexcept;

END_NAMESPACE(GShaderCache)

#endif  // #if PSYDOOM_GLES_RENDERER
