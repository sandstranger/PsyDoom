//------------------------------------------------------------------------------------------------------------------------------------------
// The OpenGL ES 3.0 renderer, core module: frame lifecycle, render paths, framebuffers, PSX VRAM and the replay of what 'VDrawing'
// recorded for the frame. Written for tile based GPUs, see the header for the details.
//------------------------------------------------------------------------------------------------------------------------------------------
#include "GRenderer.h"

#if PSYDOOM_GLES_RENDERER

#include "Asserts.h"
#include "FatalErrors.h"
#include "GShaderCache.h"
#include "Gpu.h"
#include "PsyDoom/Config/Config.h"
#include "PsyDoom/GammaTable.h"
#include "PsyDoom/PlayerPrefs.h"
#include "PsyDoom/PsxVm.h"
#include "PsyDoom/Video.h"
#include "PsyDoom/Vulkan/VDrawing.h"
#include "PsyDoom/Vulkan/VRenderer.h"
#include "PsyDoom/Vulkan/VTypes.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <SDL.h>
#include <string>
#include <vector>

BEGIN_NAMESPACE(GRenderer)

//------------------------------------------------------------------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------------------------------------------------------------------
static constexpr uint32_t   NUM_FRAME_HISTORY   = 2;        // Frames of the main render path kept (crossfades, loading plaques): as Vulkan
static constexpr uint32_t   NUM_PSX_FB_TEXTURES = 2;        // PSX framebuffer textures, alternated so none is written while being read
static constexpr uint32_t   NUM_STREAM_SLOTS    = 3;        // Ring of per frame vertex and uniform buffers (frames the GPU may be behind)
static constexpr uint32_t   NUM_STAGING_TEX     = 2;        // Staging textures for PSX VRAM updates, alternated per frame
static constexpr GLsizei    STAGING_TEX_W       = 1024;     // Size of the staging textures for PSX VRAM updates (1 MiB each)
static constexpr GLsizei    STAGING_TEX_H       = 512;
static constexpr size_t     MAX_PENDING_VRAM_RECTS = 1024;  // Past this many pending VRAM updates, VRAM is uploaded as a whole instead
static constexpr GLuint     TEX_UNIT_UPLOAD     = 7;        // Texture unit for creating and uploading textures: no unit used to draw is disturbed
static constexpr GLuint64   FENCE_WAIT_TIMEOUT  = 100'000'000;  // 100 ms per wait on a fence (waits repeat until done)

//------------------------------------------------------------------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------------------------------------------------------------------

// How MSAA is done for the main render path
enum class MsaaMode : uint8_t {
    None,
    RenderToTextureExt,     // EXT_multisampled_render_to_texture: samples only in tile memory, resolved as tiles are written out
    RenderToTextureImg,     // IMG_multisampled_render_to_texture: the same (PowerVR)
    Renderbuffer            // Multisampled renderbuffer resolved by a blit, then invalidated (GPUs without the above)
};

enum class BlendMode : uint8_t {
    None,
    Alpha,
    Additive,
    Subtractive,
    Unknown
};

// The GL state for one of the pipelines of the Vulkan renderer's main render path
struct PipelineState {
    GProgram    program;
    BlendMode   blendMode;
    bool        bCullBackFaces;
    bool        bLines;
    bool        bPerSampleShading;      // Sample rate shading when MSAA is on (as with Vulkan: the 3D world, but not the UI or sky)
};

// Uniform block of the draw programs, std140 layout ('VShaderUniforms_Draw' + padding to the block size)
struct DrawUniformBlock {
    float   mvpMatrix[16];
    float   ndcToPsxScale[2];
    float   psxNdcOffset[2];
    float   invDrawRes[2];
    float   padding[2];
};

static_assert(sizeof(DrawUniformBlock) == 96);
static_assert(sizeof(Matrix4f) == sizeof(float) * 16);

// One slot of the ring of per frame streaming buffers
struct StreamSlot {
    GLuint      vbo;
    GLuint      ubo;
    GLuint      vao;
    GLsizeiptr  vboSize;
    GLsizeiptr  uboSize;
    GLsync      fence;      // Signalled once the GPU is done with the frame that used the slot
};

// A pending update to an area of PSX VRAM
struct VramRect {
    uint16_t x, y, w, h;
};

// A copy from a staging texture to PSX VRAM
struct VramBlit {
    GLint srcX, srcY, dstX, dstY, w, h;
};

// Cache of GL state, to skip redundant state changes
struct StateCache {
    GLuint      program;
    BlendMode   blendMode;
    int8_t      bCullFace;
    int8_t      bSampleShading;
    int8_t      bScissorTest;
};

//------------------------------------------------------------------------------------------------------------------------------------------
// State
//------------------------------------------------------------------------------------------------------------------------------------------
bool gbSkipNextFramePresent;        // Draw the next (or current) frame, but don't show it
bool gbSkipNextFrame;               // Neither draw nor show the next (or current) frame
bool gbUsingGammaAdjustThisFrame;   // Whether gamma adjustment is used this frame (decided at the start of the frame)

static SDL_Window*  gpWindow;
static bool         gbInitialized;
static bool         gbAppInBackground;
static bool         gbIsRendering;
static RenderPath   gCurRenderPath = RenderPath::Main;
static RenderPath   gNextRenderPath = RenderPath::Main;

// What the device can do and what was chosen
static GLint        gMaxTextureSize;
static GLint        gUboOffsetAlignment;
static MsaaMode     gMsaaMode;
static GLsizei      gNumSamples;            // Samples of the main render path, '1' without MSAA
static bool         gbPerSampleShading;     // Sample rate shading for the 3D world when MSAA is on (OES_sample_shading)
static GLenum       gDrawColorFormat;       // Color format of the frames drawn: 16-bit like the PSX, or 32-bit
static bool         gbStagedVramUpdates;    // Whether PSX VRAM updates are copied in by the GPU (it can be a color attachment)

// Size of the screen (window surface) in pixels
static uint32_t     gScreenW;
static uint32_t     gScreenH;

// Main render path: the frames drawn (the latest ones are kept for crossfades and loading plaques).
// With EXT_multisampled_render_to_texture the framebuffers render multisampled to the single sample textures directly.
static GLuint       gHistoryTex[NUM_FRAME_HISTORY];
static GLuint       gHistoryFbo[NUM_FRAME_HISTORY];
static bool         gbRenderedHistory[NUM_FRAME_HISTORY];
static uint32_t     gHistoryIdx;                            // Which frame the next frame of the main render path is drawn to
static uint32_t     gHistoryW;
static uint32_t     gHistoryH;
static GLuint       gMsaaRenderbuffer;                      // MSAA with a renderbuffer only
static GLuint       gMsaaFbo;

// PSX render path: the PSX framebuffer, as is (R16UI)
static GLuint       gPsxFbTex[NUM_PSX_FB_TEXTURES];
static uint32_t     gPsxFbTexIdx;

// PSX VRAM, its pending updates and the staging textures used to copy them in
static GLuint                   gVramTex;
static GLuint                   gVramFbo;
static GLuint                   gStagingTex[NUM_STAGING_TEX];
static GLuint                   gStagingFbo[NUM_STAGING_TEX];
static uint32_t                 gStagingIdx;
static std::vector<VramRect>    gPendingVramRects;
static bool                     gbVramFullUploadPending;
static std::vector<VramBlit>    gVramBlits;

// Gamma adjustment remap table (256x1, R8)
static GLuint gGammaLutTex;

// Samplers for the post programs (textures sampled with normalized coordinates)
static GLuint gSamplerNearest;
static GLuint gSamplerLinear;

// Streaming buffers for the frames of the main render path, and an empty VAO for draws with no vertex attributes
static StreamSlot   gStreamSlots[NUM_STREAM_SLOTS];
static uint32_t     gStreamSlotIdx;
static GLuint       gEmptyVao;

// Uniform data for the current frame, as uploaded
static std::vector<std::byte> gUniformUploadBuffer;

static StateCache gStateCache;

//------------------------------------------------------------------------------------------------------------------------------------------
// GL state changes, skipping redundant ones
//------------------------------------------------------------------------------------------------------------------------------------------
static void invalidateStateCache() noexcept {
    gStateCache.program = UINT32_MAX;
    gStateCache.blendMode = BlendMode::Unknown;
    gStateCache.bCullFace = -1;
    gStateCache.bSampleShading = -1;
    gStateCache.bScissorTest = -1;
}

static void useProgram(const GLuint program) noexcept {
    if (gStateCache.program != program) {
        glUseProgram(program);
        gStateCache.program = program;
    }
}

static void setBlendMode(const BlendMode blendMode) noexcept {
    if (gStateCache.blendMode == blendMode)
        return;

    // As the Vulkan renderer's blend states: color per mode, alpha always 'src + dst'
    if (blendMode == BlendMode::None) {
        glDisable(GL_BLEND);
    } else {
        if ((gStateCache.blendMode == BlendMode::None) || (gStateCache.blendMode == BlendMode::Unknown)) {
            glEnable(GL_BLEND);
        }

        switch (blendMode) {
            case BlendMode::Alpha:
                glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
                glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
                break;

            case BlendMode::Additive:
                glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
                glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
                break;

            case BlendMode::Subtractive:
                glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
                glBlendEquationSeparate(GL_FUNC_REVERSE_SUBTRACT, GL_FUNC_ADD);
                break;

            default:
                break;
        }
    }

    gStateCache.blendMode = blendMode;
}

static void setCullFace(const bool bCullFace) noexcept {
    if (gStateCache.bCullFace != (int8_t) bCullFace) {
        if (bCullFace) {
            glEnable(GL_CULL_FACE);
        } else {
            glDisable(GL_CULL_FACE);
        }

        gStateCache.bCullFace = bCullFace;
    }
}

static void setSampleShading(const bool bSampleShading) noexcept {
    if (gStateCache.bSampleShading != (int8_t) bSampleShading) {
        if (bSampleShading) {
            glEnable(GL_SAMPLE_SHADING_OES);
            glMinSampleShadingOES(1.0f);
        } else {
            glDisable(GL_SAMPLE_SHADING_OES);
        }

        gStateCache.bSampleShading = bSampleShading;
    }
}

static void setScissorTest(const bool bScissorTest) noexcept {
    if (gStateCache.bScissorTest != (int8_t) bScissorTest) {
        if (bScissorTest) {
            glEnable(GL_SCISSOR_TEST);
        } else {
            glDisable(GL_SCISSOR_TEST);
        }

        gStateCache.bScissorTest = bScissorTest;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Waits until the GPU has signalled a fence and deletes it
//------------------------------------------------------------------------------------------------------------------------------------------
static void waitForAndDeleteFence(GLsync& fence) noexcept {
    if (!fence)
        return;

    while (true) {
        const GLenum result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, FENCE_WAIT_TIMEOUT);

        if (result != GL_TIMEOUT_EXPIRED)
            break;  // Signalled, or failed (context lost): either way done
    }

    glDeleteSync(fence);
    fence = nullptr;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Texture helpers
//------------------------------------------------------------------------------------------------------------------------------------------
void bindTextureForUpload(const GLuint texture) noexcept {
    glActiveTexture(GL_TEXTURE0 + TEX_UNIT_UPLOAD);
    glBindTexture(GL_TEXTURE_2D, texture);
}

static GLuint createTexture2d(const GLenum internalFormat, const GLsizei width, const GLsizei height) noexcept {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    bindTextureForUpload(texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, width, height);

    // Note: integer textures are only complete with nearest filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

static void deleteTexture(GLuint& texture, const bool bContextLost) noexcept {
    if (texture && (!bContextLost)) {
        glDeleteTextures(1, &texture);
    }

    texture = 0;
}

static void deleteFramebuffer(GLuint& fbo, const bool bContextLost) noexcept {
    if (fbo && (!bContextLost)) {
        glDeleteFramebuffers(1, &fbo);
    }

    fbo = 0;
}

static void deleteRenderbuffer(GLuint& renderbuffer, const bool bContextLost) noexcept {
    if (renderbuffer && (!bContextLost)) {
        glDeleteRenderbuffers(1, &renderbuffer);
    }

    renderbuffer = 0;
}

// Makes a framebuffer drawing to the given texture, with the given MSAA mode and sample count
static GLuint createTextureFbo(const GLuint texture, const MsaaMode msaaMode, const GLsizei numSamples) noexcept {
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    if ((msaaMode == MsaaMode::RenderToTextureExt) && (numSamples > 1)) {
        glFramebufferTexture2DMultisampleEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0, numSamples);
    } else if ((msaaMode == MsaaMode::RenderToTextureImg) && (numSamples > 1)) {
        glFramebufferTexture2DMultisampleIMG(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0, numSamples);
    } else {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    }

    return fbo;
}

static bool isBoundFramebufferComplete() noexcept {
    return (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Device capabilities: decides the color format of the frames drawn and how MSAA is done
//------------------------------------------------------------------------------------------------------------------------------------------

// Tells if frames can be drawn with the given format and MSAA, by making a small framebuffer
static bool canDrawWith(const GLenum colorFormat, const MsaaMode msaaMode, const GLsizei numSamples) noexcept {
    const GLuint texture = createTexture2d(colorFormat, 64, 64);
    GLuint fbo = 0;
    GLuint renderbuffer = 0;
    bool bComplete = false;

    if (msaaMode == MsaaMode::Renderbuffer) {
        glGenRenderbuffers(1, &renderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, numSamples, colorFormat, 64, 64);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
        bComplete = isBoundFramebufferComplete();

        // The texture the samples are resolved to must also be a valid color attachment
        if (bComplete) {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
            bComplete = isBoundFramebufferComplete();
        }
    } else {
        fbo = createTextureFbo(texture, msaaMode, numSamples);
        bComplete = isBoundFramebufferComplete();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    deleteFramebuffer(fbo, false);
    deleteRenderbuffer(renderbuffer, false);
    GLuint textureToDelete = texture;
    deleteTexture(textureToDelete, false);
    glGetError();   // Clear any errors from formats or sample counts that are not supported
    return bComplete;
}

// The most samples a format supports for a multisampled renderbuffer
static GLsizei getMaxRenderbufferSamples(const GLenum colorFormat) noexcept {
    GLint numSampleCounts = 0;
    glGetInternalformativ(GL_RENDERBUFFER, colorFormat, GL_NUM_SAMPLE_COUNTS, 1, &numSampleCounts);

    if (numSampleCounts <= 0)
        return 1;

    std::vector<GLint> sampleCounts((size_t) numSampleCounts);
    glGetInternalformativ(GL_RENDERBUFFER, colorFormat, GL_SAMPLES, numSampleCounts, sampleCounts.data());
    return std::max(*std::max_element(sampleCounts.begin(), sampleCounts.end()), 1);   // Note: listed in descending order, but be safe
}

static void decideDrawFormatAndMsaa() noexcept {
    // Formats to try: 16-bit color like the Vulkan renderer (the PSX's precision when blending) unless 32-bit shading is wanted.
    // Dithering is turned off (see 'initGlState'): it would change the 16-bit color, which Vulkan does not do.
    std::vector<GLenum> colorFormats;

    if (!Config::gbUseVulkan32BitShading) {
        colorFormats.push_back(GL_RGB5_A1);
    }

    colorFormats.push_back(GL_RGBA8);

    // MSAA mode: the tile based one where available
    const int32_t wantedSamples = std::max(Config::gAAMultisamples, 1);
    MsaaMode msaaMode = MsaaMode::None;
    GLint maxSamples = 1;

    if (wantedSamples > 1) {
        if (GLAD_GL_EXT_multisampled_render_to_texture && glFramebufferTexture2DMultisampleEXT) {
            msaaMode = MsaaMode::RenderToTextureExt;
            glGetIntegerv(GL_MAX_SAMPLES_EXT, &maxSamples);
        } else if (GLAD_GL_IMG_multisampled_render_to_texture && glFramebufferTexture2DMultisampleIMG) {
            msaaMode = MsaaMode::RenderToTextureImg;
            glGetIntegerv(GL_MAX_SAMPLES_IMG, &maxSamples);
        } else {
            msaaMode = MsaaMode::Renderbuffer;
            glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
        }
    }

    // Try each format, with the most samples wanted and possible first, then fewer: the first combination that works is used.
    // MSAA is preferred over 16-bit color.
    for (const GLenum colorFormat : colorFormats) {
        GLsizei maxSamplesForFormat = std::max(maxSamples, 1);

        if (msaaMode == MsaaMode::Renderbuffer) {
            maxSamplesForFormat = std::min(maxSamplesForFormat, getMaxRenderbufferSamples(colorFormat));
        }

        GLsizei numSamples = 1;

        while ((numSamples * 2 <= wantedSamples) && (numSamples * 2 <= maxSamplesForFormat)) {
            numSamples *= 2;
        }

        for (; numSamples > 1; numSamples /= 2) {
            if (canDrawWith(colorFormat, msaaMode, numSamples)) {
                gDrawColorFormat = colorFormat;
                gMsaaMode = msaaMode;
                gNumSamples = numSamples;
                return;
            }
        }
    }

    // No MSAA (or not possible): the first format that works
    for (const GLenum colorFormat : colorFormats) {
        if (canDrawWith(colorFormat, MsaaMode::None, 1)) {
            gDrawColorFormat = colorFormat;
            gMsaaMode = MsaaMode::None;
            gNumSamples = 1;
            return;
        }
    }

    FatalErrors::raise("GLES renderer: no color format can be drawn to!");
}

static void determineCaps() noexcept {
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gMaxTextureSize);
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &gUboOffsetAlignment);
    gUboOffsetAlignment = std::max(gUboOffsetAlignment, 1);

    gDrawColorFormat = 0;
    gMsaaMode = MsaaMode::None;
    gNumSamples = 1;
    decideDrawFormatAndMsaa();

    // Sample rate shading as with Vulkan: the textures antialiased too, unless only the edges are wanted ('AntiAliasingEdgesOnly').
    // MSAA itself is nearly free on tile based GPUs, sample rate shading multiplies the fragment shading work by the sample count.
    gbPerSampleShading = (
        (gNumSamples > 1) &&
        (!Config::gbAAEdgesOnly) &&
        GLAD_GL_OES_sample_shading &&
        (glMinSampleShadingOES != nullptr)
    );

    const char* const msaaModeStr = (gMsaaMode == MsaaMode::RenderToTextureExt) ? "EXT_multisampled_render_to_texture" :
        (gMsaaMode == MsaaMode::RenderToTextureImg) ? "IMG_multisampled_render_to_texture" :
        (gMsaaMode == MsaaMode::Renderbuffer) ? "renderbuffer + blit" : "off";

    SDL_Log(
        "PsyDoom GLES: %s | %s | %s | color %s | MSAA %dx (%s, wanted %dx)%s",
        (const char*) glGetString(GL_VENDOR),
        (const char*) glGetString(GL_RENDERER),
        (const char*) glGetString(GL_VERSION),
        (gDrawColorFormat == GL_RGB5_A1) ? "16-bit" : "32-bit",
        (int) gNumSamples,
        msaaModeStr,
        (int) std::max(Config::gAAMultisamples, 1),
        (gbPerSampleShading) ? ", sample rate shading" : ""
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Main render path: the frames drawn
//------------------------------------------------------------------------------------------------------------------------------------------
static void destroyFrameHistory(const bool bContextLost) noexcept {
    for (uint32_t i = 0; i < NUM_FRAME_HISTORY; ++i) {
        deleteFramebuffer(gHistoryFbo[i], bContextLost);
        deleteTexture(gHistoryTex[i], bContextLost);
        gbRenderedHistory[i] = false;
    }

    deleteFramebuffer(gMsaaFbo, bContextLost);
    deleteRenderbuffer(gMsaaRenderbuffer, bContextLost);
    gHistoryIdx = 0;
    gHistoryW = 0;
    gHistoryH = 0;
}

// Makes the frames drawn by the main render path if their size changed: returns 'false' on failure
static bool ensureFrameHistory() noexcept {
    const uint32_t fbW = VRenderer::gFramebufferW;
    const uint32_t fbH = VRenderer::gFramebufferH;

    if ((fbW == 0) || (fbH == 0))
        return false;

    if (gHistoryTex[0] && (gHistoryW == fbW) && (gHistoryH == fbH))
        return true;

    destroyFrameHistory(false);
    gHistoryW = fbW;
    gHistoryH = fbH;

    bool bAllComplete = true;

    for (uint32_t i = 0; i < NUM_FRAME_HISTORY; ++i) {
        gHistoryTex[i] = createTexture2d(gDrawColorFormat, (GLsizei) fbW, (GLsizei) fbH);
        gHistoryFbo[i] = createTextureFbo(gHistoryTex[i], gMsaaMode, gNumSamples);
        bAllComplete &= isBoundFramebufferComplete();

        // Start black: a crossfade or loading plaque may show a frame never drawn
        setScissorTest(false);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    if (gMsaaMode == MsaaMode::Renderbuffer) {
        glGenRenderbuffers(1, &gMsaaRenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, gMsaaRenderbuffer);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, gNumSamples, gDrawColorFormat, (GLsizei) fbW, (GLsizei) fbH);
        glGenFramebuffers(1, &gMsaaFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, gMsaaFbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gMsaaRenderbuffer);
        bAllComplete &= isBoundFramebufferComplete();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (!bAllComplete) {
        SDL_Log("PsyDoom GLES: failed to make the framebuffers for a %ux%u frame!", fbW, fbH);
        destroyFrameHistory(false);
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// PSX VRAM
//------------------------------------------------------------------------------------------------------------------------------------------

// Makes sure PSX VRAM fits in the largest texture the device can do, otherwise makes it smaller (as the Vulkan renderer does)
static void fitPsxVramToMaxTextureSize() noexcept {
    Gpu::Core& psxGpu = PsxVm::gGpu;
    uint32_t vramW = psxGpu.ramPixelW;
    uint32_t vramH = psxGpu.ramPixelH;

    while (((vramW > (uint32_t) gMaxTextureSize) || (vramH > (uint32_t) gMaxTextureSize)) && ((vramW > 1024) || (vramH > 512))) {
        if ((vramH > vramW) || (vramW <= 1024)) {
            vramH /= 2;
        } else {
            vramW /= 2;
        }
    }

    if ((vramW != psxGpu.ramPixelW) || (vramH != psxGpu.ramPixelH)) {
        std::printf(
            "PsyDoom: WARNING: due to max texture size restrictions, the GLES device can't support the desired VRAM size of %ux%u.\n"
            "PsyDoom will use %ux%u instead! To silence this warning, set a smaller 'VramSizeInMegabytes' in 'graphics_cfg.ini'.\n",
            (unsigned) psxGpu.ramPixelW, (unsigned) psxGpu.ramPixelH, vramW, vramH
        );

        Gpu::destroyCore(psxGpu);
        Gpu::initCore(psxGpu, (uint16_t) vramW, (uint16_t) vramH);
    }
}

// Uploads an area of PSX VRAM straight from the PSX GPU's memory to the texture
static void uploadVramRectDirect(const GLuint texture, const GLint dstX, const GLint dstY, const VramRect& rect) noexcept {
    const Gpu::Core& psxGpu = PsxVm::gGpu;
    bindTextureForUpload(texture);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, psxGpu.ramPixelW);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, rect.x);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, rect.y);
    glTexSubImage2D(GL_TEXTURE_2D, 0, dstX, dstY, rect.w, rect.h, GL_RED_INTEGER, GL_UNSIGNED_SHORT, psxGpu.pRam);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
}

static void initVram() noexcept {
    Gpu::Core& psxGpu = PsxVm::gGpu;
    gVramTex = createTexture2d(GL_R16UI, psxGpu.ramPixelW, psxGpu.ramPixelH);
    uploadVramRectDirect(gVramTex, 0, 0, VramRect{ 0, 0, psxGpu.ramPixelW, psxGpu.ramPixelH });
    gPendingVramRects.clear();
    gbVramFullUploadPending = false;

    // VRAM as a color attachment, for the GPU to copy updates into it from the staging textures
    glGenFramebuffers(1, &gVramFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, gVramFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gVramTex, 0);
    gbStagedVramUpdates = isBoundFramebufferComplete();

    for (uint32_t i = 0; (i < NUM_STAGING_TEX) && gbStagedVramUpdates; ++i) {
        gStagingTex[i] = createTexture2d(GL_R16UI, STAGING_TEX_W, STAGING_TEX_H);
        gStagingFbo[i] = createTextureFbo(gStagingTex[i], MsaaMode::None, 1);
        gbStagedVramUpdates &= isBoundFramebufferComplete();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    gStagingIdx = 0;

    if (!gbStagedVramUpdates) {
        SDL_Log("PsyDoom GLES: PSX VRAM can't be a color attachment, updates to it are uploaded directly");
    }

    // VRAM stays bound to its texture unit for good
    glActiveTexture(GL_TEXTURE0 + GShaders::TEX_UNIT_VRAM);
    glBindTexture(GL_TEXTURE_2D, gVramTex);
}

static void destroyVram(const bool bContextLost) noexcept {
    for (uint32_t i = 0; i < NUM_STAGING_TEX; ++i) {
        deleteFramebuffer(gStagingFbo[i], bContextLost);
        deleteTexture(gStagingTex[i], bContextLost);
    }

    deleteFramebuffer(gVramFbo, bContextLost);
    deleteTexture(gVramTex, bContextLost);
    gPendingVramRects.clear();
    gVramBlits.clear();
    gbVramFullUploadPending = false;
    gbStagedVramUpdates = false;
}

// Copies the areas placed in the current staging texture into VRAM, on the GPU
static void flushVramBlits() noexcept {
    if (gVramBlits.empty())
        return;

    setScissorTest(false);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, gStagingFbo[gStagingIdx]);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gVramFbo);

    for (const VramBlit& blit : gVramBlits) {
        glBlitFramebuffer(
            blit.srcX, blit.srcY, blit.srcX + blit.w, blit.srcY + blit.h,
            blit.dstX, blit.dstY, blit.dstX + blit.w, blit.dstY + blit.h,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST
        );
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    gVramBlits.clear();
}

// Applies all of the PSX VRAM updates made since the last frame (main or PSX render path), before anything reads VRAM.
// The areas are packed into the staging texture (rows of areas) and copied into VRAM by the GPU, in order with the frames that read it.
static void applyPendingVramUpdates() noexcept {
    if (gbVramFullUploadPending) {
        // Rare (many updates while not on the main render path): upload everything as is
        const Gpu::Core& psxGpu = PsxVm::gGpu;
        uploadVramRectDirect(gVramTex, 0, 0, VramRect{ 0, 0, psxGpu.ramPixelW, psxGpu.ramPixelH });
        gPendingVramRects.clear();
        gbVramFullUploadPending = false;
        return;
    }

    if (gPendingVramRects.empty())
        return;

    if (!gbStagedVramUpdates) {
        for (const VramRect& rect : gPendingVramRects) {
            uploadVramRectDirect(gVramTex, rect.x, rect.y, rect);
        }

        gPendingVramRects.clear();
        return;
    }

    GLint penX = 0;
    GLint penY = 0;
    GLint rowH = 0;

    for (const VramRect& rect : gPendingVramRects) {
        // Areas bigger than the staging texture are done in pieces
        for (uint16_t pieceY = 0; pieceY < rect.h; pieceY += (uint16_t) STAGING_TEX_H) {
            for (uint16_t pieceX = 0; pieceX < rect.w; pieceX += (uint16_t) STAGING_TEX_W) {
                const VramRect piece = {
                    (uint16_t)(rect.x + pieceX),
                    (uint16_t)(rect.y + pieceY),
                    (uint16_t) std::min<GLint>(rect.w - pieceX, STAGING_TEX_W),
                    (uint16_t) std::min<GLint>(rect.h - pieceY, STAGING_TEX_H)
                };

                // Place it in the current row of the staging texture, or a new row, or the next staging texture if full
                if (penX + piece.w > STAGING_TEX_W) {
                    penX = 0;
                    penY += rowH;
                    rowH = 0;
                }

                if (penY + piece.h > STAGING_TEX_H) {
                    flushVramBlits();
                    gStagingIdx = (gStagingIdx + 1) % NUM_STAGING_TEX;
                    penX = 0;
                    penY = 0;
                    rowH = 0;
                }

                uploadVramRectDirect(gStagingTex[gStagingIdx], penX, penY, piece);
                gVramBlits.push_back(VramBlit{ penX, penY, piece.x, piece.y, piece.w, piece.h });
                penX += piece.w;
                rowH = std::max<GLint>(rowH, piece.h);
            }
        }
    }

    flushVramBlits();
    gStagingIdx = (gStagingIdx + 1) % NUM_STAGING_TEX;  // The next frame's updates go to the other staging texture
    gPendingVramRects.clear();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Streaming buffers (vertices and uniforms of the frames of the main render path)
//------------------------------------------------------------------------------------------------------------------------------------------
static void initStreamSlots() noexcept {
    constexpr GLsizei STRIDE = (GLsizei) sizeof(VVertex_Draw);
    static_assert(sizeof(VVertex_Draw) == 40);

    for (StreamSlot& slot : gStreamSlots) {
        slot = {};
        glGenBuffers(1, &slot.vbo);
        glGenBuffers(1, &slot.ubo);
        glGenVertexArrays(1, &slot.vao);

        glBindVertexArray(slot.vao);
        glBindBuffer(GL_ARRAY_BUFFER, slot.vbo);
        slot.vboSize = 1024 * 1024;
        glBufferData(GL_ARRAY_BUFFER, slot.vboSize, nullptr, GL_STREAM_DRAW);

        // The vertex layout: 'VVertex_Draw' (see 'DECLARE_VS_INPUTS_VVERTEX_DRAW' in the Vulkan shaders)
        for (GLuint attrib = 0; attrib <= 6; ++attrib) {
            glEnableVertexAttribArray(attrib);
        }

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, x));
        glVertexAttribIPointer(1, 4, GL_UNSIGNED_BYTE, STRIDE, (const void*) offsetof(VVertex_Draw, r));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, u));
        glVertexAttribIPointer(3, 2, GL_UNSIGNED_SHORT, STRIDE, (const void*) offsetof(VVertex_Draw, texWinX));
        glVertexAttribIPointer(4, 2, GL_UNSIGNED_SHORT, STRIDE, (const void*) offsetof(VVertex_Draw, texWinW));
        glVertexAttribIPointer(5, 2, GL_UNSIGNED_SHORT, STRIDE, (const void*) offsetof(VVertex_Draw, clutX));
        glVertexAttribIPointer(6, 4, GL_UNSIGNED_BYTE, STRIDE, (const void*) offsetof(VVertex_Draw, stmulR));

        glBindBuffer(GL_UNIFORM_BUFFER, slot.ubo);
        slot.uboSize = 16 * 1024;
        glBufferData(GL_UNIFORM_BUFFER, slot.uboSize, nullptr, GL_STREAM_DRAW);
    }

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    gStreamSlotIdx = 0;
}

static void destroyStreamSlots(const bool bContextLost) noexcept {
    for (StreamSlot& slot : gStreamSlots) {
        if (!bContextLost) {
            waitForAndDeleteFence(slot.fence);
            glDeleteVertexArrays(1, &slot.vao);
            glDeleteBuffers(1, &slot.ubo);
            glDeleteBuffers(1, &slot.vbo);
        }

        slot = {};
    }

    gStreamSlotIdx = 0;
}

// Uploads data to a buffer the GPU no longer uses (its fence was waited on), growing it if needed
static void uploadToStreamBuffer(const GLenum target, const GLuint buffer, GLsizeiptr& bufferSize, const void* const pData, const GLsizeiptr size) noexcept {
    glBindBuffer(target, buffer);

    if (size > bufferSize) {
        bufferSize = std::max(size, bufferSize * 2);
        glBufferData(target, bufferSize, nullptr, GL_STREAM_DRAW);
    }

    if (size > 0) {
        glBufferSubData(target, 0, size, pData);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Main render path: replaying the frame drawn through 'VDrawing'
//------------------------------------------------------------------------------------------------------------------------------------------

// The GL state for each pipeline of the Vulkan renderer (see 'VPipelines::initPipelineSet_Main')
static PipelineState getPipelineState(const VPipelineType_Main pipelineType) noexcept {
    switch (pipelineType) {
        case VPipelineType_Main::Lines:                     return { GProgram::Colored,             BlendMode::None,        false,  true,   false };
        case VPipelineType_Main::Colored:                   return { GProgram::Colored,             BlendMode::None,        false,  false,  false };
        case VPipelineType_Main::UI_4bpp:                   return { GProgram::UI_4bpp,             BlendMode::None,        false,  false,  false };
        case VPipelineType_Main::UI_8bpp:                   return { GProgram::UI_8bpp,             BlendMode::None,        false,  false,  false };
        case VPipelineType_Main::UI_8bpp_Add:               return { GProgram::UI_8bpp,             BlendMode::Additive,    false,  false,  false };
        case VPipelineType_Main::UI_16bpp:                  return { GProgram::UI_16bpp,            BlendMode::None,        false,  false,  false };
        case VPipelineType_Main::World_GeomMasked:          return { GProgram::World_Wrap,          BlendMode::None,        true,   false,  true  };
        case VPipelineType_Main::World_GeomOpaque:          return { GProgram::World_WrapOpaque,    BlendMode::None,        true,   false,  true  };
        case VPipelineType_Main::World_GeomAlpha:           return { GProgram::World_Wrap,          BlendMode::Alpha,       true,   false,  true  };
        case VPipelineType_Main::World_SpriteMasked:        return { GProgram::World_Clamp,         BlendMode::None,        false,  false,  true  };
        case VPipelineType_Main::World_SpriteAlpha:         return { GProgram::World_Clamp,         BlendMode::Alpha,       false,  false,  true  };
        case VPipelineType_Main::World_SpriteAdditive:      return { GProgram::World_Clamp,         BlendMode::Additive,    false,  false,  true  };
        case VPipelineType_Main::World_SpriteSubtractive:   return { GProgram::World_Clamp,         BlendMode::Subtractive, false,  false,  true  };
        case VPipelineType_Main::World_Sky:                 return { GProgram::Sky,                 BlendMode::None,        true,   false,  false };

        // Not used for drawing through 'VDrawing' (MSAA resolve and gamma adjustment are done differently here)
        default:                                            return { GProgram::Colored,             BlendMode::None,        false,  false,  false };
    }
}

// Packs the frame's uniforms for upload: the first block is zeroed (bound before any uniforms are set), the frame's ones follow it
static GLsizeiptr packFrameUniforms(const VDrawing::FrameDrawData& frameData, const GLsizeiptr blockStride) noexcept {
    const GLsizeiptr totalSize = blockStride * ((GLsizeiptr) frameData.numUniforms + 1);
    gUniformUploadBuffer.assign((size_t) totalSize, std::byte(0));

    for (uint32_t i = 0; i < frameData.numUniforms; ++i) {
        const VShaderUniforms_Draw& src = frameData.pUniforms[i];
        DrawUniformBlock block = {};
        std::memcpy(block.mvpMatrix, &src.mvpMatrix, sizeof(block.mvpMatrix));
        block.ndcToPsxScale[0] = src.ndcToPsxScaleX;
        block.ndcToPsxScale[1] = src.ndcToPsxScaleY;
        block.psxNdcOffset[0] = src.psxNdcOffsetX;
        block.psxNdcOffset[1] = src.psxNdcOffsetY;
        block.invDrawRes[0] = src.invDrawResX;
        block.invDrawRes[1] = src.invDrawResY;
        std::memcpy(gUniformUploadBuffer.data() + (size_t) blockStride * (i + 1), &block, sizeof(block));
    }

    return totalSize;
}

// Draws the frame recorded by 'VDrawing' into the frame history texture at the given index
static void drawMainFrame(const VDrawing::FrameDrawData& frameData, const uint32_t historyIdx) noexcept {
    // Get the buffers for this frame, waiting until the GPU is done with the frame that used them last (normally long done)
    StreamSlot& slot = gStreamSlots[gStreamSlotIdx];
    waitForAndDeleteFence(slot.fence);

    const GLsizeiptr uniformBlockStride = ((GLsizeiptr) sizeof(DrawUniformBlock) + gUboOffsetAlignment - 1) / gUboOffsetAlignment * gUboOffsetAlignment;
    const GLsizeiptr uniformsSize = packFrameUniforms(frameData, uniformBlockStride);
    uploadToStreamBuffer(GL_UNIFORM_BUFFER, slot.ubo, slot.uboSize, gUniformUploadBuffer.data(), uniformsSize);
    uploadToStreamBuffer(GL_ARRAY_BUFFER, slot.vbo, slot.vboSize, frameData.pVerts, (GLsizeiptr) frameData.numVerts * (GLsizeiptr) sizeof(VVertex_Draw));

    // Start the pass: clear everything first, so the GPU never loads the old contents into tile memory
    const GLuint drawFbo = (gMsaaMode == MsaaMode::Renderbuffer) ? gMsaaFbo : gHistoryFbo[historyIdx];
    glBindFramebuffer(GL_FRAMEBUFFER, drawFbo);
    glViewport(0, 0, (GLsizei) gHistoryW, (GLsizei) gHistoryH);
    setScissorTest(false);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // The viewport and scissors for drawing: the same numbers as with Vulkan (the image's first row is the top in both)
    int32_t viewportX = {}, viewportY = {}, viewportW = {}, viewportH = {};
    VRenderer::getDrawViewport(viewportX, viewportY, viewportW, viewportH);
    glViewport(viewportX, viewportY, viewportW, viewportH);
    glScissor(viewportX, viewportY, viewportW, viewportH);
    setScissorTest(true);

    // Replay the drawing commands
    glBindVertexArray(slot.vao);
    glBindBufferRange(GL_UNIFORM_BUFFER, GShaders::DRAW_UNIFORMS_BINDING, slot.ubo, 0, (GLsizeiptr) sizeof(DrawUniformBlock));
    GLenum primitiveType = GL_TRIANGLES;

    for (uint32_t cmdIdx = 0; cmdIdx < frameData.numCmds; ++cmdIdx) {
        const VDrawing::DrawCmd& cmd = frameData.pCmds[cmdIdx];

        switch (cmd.type) {
            case VDrawing::DrawCmdType::SetPipeline: {
                const PipelineState state = getPipelineState((VPipelineType_Main) cmd.arg1);
                useProgram(GShaders::get(state.program).program);
                setBlendMode(state.blendMode);
                setCullFace(state.bCullBackFaces);

                if (gbPerSampleShading) {
                    setSampleShading(state.bPerSampleShading);
                }

                primitiveType = (state.bLines) ? GL_LINES : GL_TRIANGLES;
            }   break;

            case VDrawing::DrawCmdType::SetUniforms: {
                glBindBufferRange(
                    GL_UNIFORM_BUFFER,
                    GShaders::DRAW_UNIFORMS_BINDING,
                    slot.ubo,
                    uniformBlockStride * ((GLintptr) cmd.arg1 + 1),
                    (GLsizeiptr) sizeof(DrawUniformBlock)
                );
            }   break;

            case VDrawing::DrawCmdType::Draw: {
                glDrawArrays(primitiveType, (GLint) cmd.arg2, (GLsizei) cmd.arg1);
            }   break;
        }
    }

    glBindVertexArray(0);
    setBlendMode(BlendMode::None);
    setCullFace(false);

    if (gbPerSampleShading) {
        setSampleShading(false);
    }

    // MSAA with a renderbuffer: resolve into the frame's texture, then drop the samples so they are never written to memory
    if (gMsaaMode == MsaaMode::Renderbuffer) {
        setScissorTest(false);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gMsaaFbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gHistoryFbo[historyIdx]);
        glBlitFramebuffer(
            0, 0, (GLint) gHistoryW, (GLint) gHistoryH,
            0, 0, (GLint) gHistoryW, (GLint) gHistoryH,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST
        );

        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        glInvalidateFramebuffer(GL_READ_FRAMEBUFFER, 1, &attachment);
    }

    // The GPU is done with this slot's buffers once it is past this point
    slot.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    gStreamSlotIdx = (gStreamSlotIdx + 1) % NUM_STREAM_SLOTS;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Screen passes: drawing to the window surface
//------------------------------------------------------------------------------------------------------------------------------------------

// Starts drawing to the window surface: cleared first, so the GPU never loads its old contents into tile memory
static void beginScreenPass() noexcept {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, (GLsizei) gScreenW, (GLsizei) gScreenH);
    setScissorTest(false);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

// Unbinds the textures of the post programs once drawn: frame textures are drawn to later, and some drivers treat a texture bound while
// it is drawn to as a feedback loop, even if no program reads it
static void unbindPostTextures() noexcept {
    for (const GLuint unit : { GShaders::TEX_UNIT_SRC, GShaders::TEX_UNIT_SRC2 }) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

static void bindPostTexture(const GLuint unit, const GLuint texture, const bool bLinearFilter) noexcept {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindSampler(unit, (bLinearFilter) ? gSamplerLinear : gSamplerNearest);
}

void drawFullscreenPost(const GProgram program, const GLuint srcTex, const GLuint srcTex2, const float lerp) noexcept {
    const GProgramInfo& programInfo = GShaders::get(program);
    useProgram(programInfo.program);
    setBlendMode(BlendMode::None);
    setCullFace(false);

    if (programInfo.uLerp >= 0) {
        glUniform1f(programInfo.uLerp, lerp);
    }

    bindPostTexture(GShaders::TEX_UNIT_SRC, srcTex, false);

    if (srcTex2) {
        bindPostTexture(GShaders::TEX_UNIT_SRC2, srcTex2, false);
    }

    glBindVertexArray(gEmptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    unbindPostTextures();
}

void drawQuad(
    const GProgram program,
    const float posRect[4],
    const float uvRect[4],
    const GLuint texture,
    const bool bLinearFilter
) noexcept {
    const GProgramInfo& programInfo = GShaders::get(program);
    useProgram(programInfo.program);
    setBlendMode(BlendMode::None);
    setCullFace(false);
    glUniform4f(programInfo.uPosRect, posRect[0], posRect[1], posRect[2], posRect[3]);
    glUniform4f(programInfo.uUvRect, uvRect[0], uvRect[1], uvRect[2], uvRect[3]);

    // Note: integer textures (the PSX framebuffer) must not be filtered
    bindPostTexture(GShaders::TEX_UNIT_SRC, texture, bLinearFilter);

    glBindVertexArray(gEmptyVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    unbindPostTextures();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// End of frame for each render path
//------------------------------------------------------------------------------------------------------------------------------------------
static void endFrame_Main(const bool bSkipPresent) noexcept {
    const VDrawing::FrameDrawData frameData = VDrawing::getFrameDrawDataGles();

    // Pass 1: PSX VRAM updates, before anything reads VRAM
    applyPendingVramUpdates();

    if (!gbSkipNextFrame) {
        // Pass 2: the frame
        const uint32_t historyIdx = gHistoryIdx;
        drawMainFrame(frameData, historyIdx);
        gbRenderedHistory[historyIdx] = true;
        gHistoryIdx = (historyIdx + 1) % NUM_FRAME_HISTORY;

        // Pass 3: the frame on the screen (scaled to it if the render resolution is not the screen's), with gamma adjustment
        if (!bSkipPresent) {
            beginScreenPass();
            const GProgram program = (gbUsingGammaAdjustThisFrame) ? GProgram::Present_Gamma : GProgram::Present;
            drawFullscreenPost(program, gHistoryTex[historyIdx], 0, 0.0f);
        }
    }

    VDrawing::endFrameGles();
}

static void endFrame_Psx(const bool bSkipPresent) noexcept {
    // Keep PSX VRAM up to date on this render path too (before the screen pass: no framebuffer switch in the middle of it)
    applyPendingVramUpdates();

    // Upload the PSX framebuffer as is: the shader decodes the PSX 16-bit pixels (no conversion on the CPU).
    // Alternate between textures, so a texture a frame in flight still reads is never written.
    const Gpu::Core& psxGpu = PsxVm::gGpu;
    const GLuint psxFbTex = gPsxFbTex[gPsxFbTexIdx];
    gPsxFbTexIdx = (gPsxFbTexIdx + 1) % NUM_PSX_FB_TEXTURES;

    const VramRect displayRect = { psxGpu.displayAreaX, psxGpu.displayAreaY, Video::ORIG_DRAW_RES_X, Video::ORIG_DRAW_RES_Y };
    uploadVramRectDirect(psxFbTex, 0, 0, displayRect);

    if (gbSkipNextFrame || bSkipPresent)
        return;

    // Where to show it: the same area as the Vulkan renderer
    float blitDstX_f = {}, blitDstY_f = {}, blitDstW_f = {}, blitDstH_f = {};
    Video::getClassicFramebufferWindowRect((float) gScreenW, (float) gScreenH, blitDstX_f, blitDstY_f, blitDstW_f, blitDstH_f);

    const int32_t blitDstMinX = std::clamp<int32_t>((int32_t) blitDstX_f, 0, (int32_t) gScreenW);
    const int32_t blitDstMinY = std::clamp<int32_t>((int32_t) blitDstY_f, 0, (int32_t) gScreenH);
    const int32_t blitDstMaxX = std::clamp<int32_t>((int32_t)(blitDstX_f + std::ceil(blitDstW_f)), 0, (int32_t) gScreenW);
    const int32_t blitDstMaxY = std::clamp<int32_t>((int32_t)(blitDstY_f + std::ceil(blitDstH_f)), 0, (int32_t) gScreenH);

    beginScreenPass();

    if ((blitDstMaxX > blitDstMinX) && (blitDstMaxY > blitDstMinY)) {
        // Note: OpenGL NDC has y up, rows are counted from the top here
        const float posRect[4] = {
            ((float) blitDstMinX / (float) gScreenW) * 2.0f - 1.0f,
            1.0f - ((float) blitDstMinY / (float) gScreenH) * 2.0f,
            ((float) blitDstMaxX / (float) gScreenW) * 2.0f - 1.0f,
            1.0f - ((float) blitDstMaxY / (float) gScreenH) * 2.0f
        };

        const float uvRect[4] = {
            0.0f,
            (float) Video::gTopOverscan / (float) Video::ORIG_DRAW_RES_Y,
            1.0f,
            (float)(Video::ORIG_DRAW_RES_Y - Video::gBotOverscan) / (float) Video::ORIG_DRAW_RES_Y
        };

        const GProgram program = (gbUsingGammaAdjustThisFrame) ? GProgram::PsxPresent_Gamma : GProgram::PsxPresent;
        drawQuad(program, posRect, uvRect, psxFbTex, false);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// GL resources: made on startup and remade if the GL context is lost
//------------------------------------------------------------------------------------------------------------------------------------------
static void initGlState() noexcept {
    invalidateStateCache();
    glDisable(GL_DITHER);           // On by default in OpenGL: would change the 16-bit color, which the Vulkan renderer doesn't do
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glFrontFace(GL_CCW);            // Vulkan's clockwise front faces: the winding is the other way around in OpenGL (y up window space)
    glCullFace(GL_BACK);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    setBlendMode(BlendMode::None);
    setCullFace(false);
    setScissorTest(false);

    if (gbPerSampleShading) {
        setSampleShading(false);
    }
}

static void initGlResources() noexcept {
    initGlState();

    // Shaders: binaries from the cache where possible ('glProgramBinary' needs at least one binary format)
    GLint numBinaryFormats = 0;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &numBinaryFormats);
    const bool bUseProgramBinaries = (numBinaryFormats > 0);

    const std::string driverId = std::string((const char*) glGetString(GL_VENDOR)) + "|" +
        (const char*) glGetString(GL_RENDERER) + "|" +
        (const char*) glGetString(GL_VERSION) + "|" +
        (const char*) glGetString(GL_SHADING_LANGUAGE_VERSION);

    GShaderCache::init(driverId, bUseProgramBinaries);
    GShaders::init(!Config::gbUseVulkan32BitShading, bUseProgramBinaries);

    // Samplers for the post programs
    glGenSamplers(1, &gSamplerNearest);
    glSamplerParameteri(gSamplerNearest, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(gSamplerNearest, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(gSamplerNearest, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(gSamplerNearest, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenSamplers(1, &gSamplerLinear);
    glSamplerParameteri(gSamplerLinear, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(gSamplerLinear, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(gSamplerLinear, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(gSamplerLinear, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Textures, buffers and framebuffers
    initVram();
    rebuildGammaAdjustTex();

    for (GLuint& texture : gPsxFbTex) {
        texture = createTexture2d(GL_R16UI, Video::ORIG_DRAW_RES_X, Video::ORIG_DRAW_RES_Y);
    }

    gPsxFbTexIdx = 0;
    initStreamSlots();
    glGenVertexArrays(1, &gEmptyVao);

    // The frame history is made at the start of the next frame (needs the screen size)
    gScreenW = 0;
    gScreenH = 0;
}

static void destroyGlResources(const bool bContextLost) noexcept {
    destroyFrameHistory(bContextLost);
    destroyStreamSlots(bContextLost);

    if (gEmptyVao && (!bContextLost)) {
        glDeleteVertexArrays(1, &gEmptyVao);
    }

    gEmptyVao = 0;

    for (GLuint& texture : gPsxFbTex) {
        deleteTexture(texture, bContextLost);
    }

    deleteTexture(gGammaLutTex, bContextLost);
    destroyVram(bContextLost);

    if (!bContextLost) {
        if (gSamplerLinear) {
            glDeleteSamplers(1, &gSamplerLinear);
        }

        if (gSamplerNearest) {
            glDeleteSamplers(1, &gSamplerNearest);
        }
    }

    gSamplerLinear = 0;
    gSamplerNearest = 0;
    GShaders::destroy(bContextLost);
    gUniformUploadBuffer.clear();
    gUniformUploadBuffer.shrink_to_fit();
    gScreenW = 0;
    gScreenH = 0;
    invalidateStateCache();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the renderer: the GL context must be current and the GL functions loaded
//------------------------------------------------------------------------------------------------------------------------------------------
bool init(SDL_Window* const pWindow) noexcept {
    ASSERT(pWindow);
    ASSERT(!gbInitialized);

    gpWindow = pWindow;
    gbAppInBackground = false;
    gbIsRendering = false;
    gbSkipNextFrame = false;
    gbSkipNextFramePresent = false;

    determineCaps();
    fitPsxVramToMaxTextureSize();
    VRenderer::updateCoordSysInfo(0, 0);    // Invalid until the first frame
    VDrawing::initForGles();
    initGlResources();

    // Start on the same render path as the Vulkan renderer would
    gCurRenderPath = (PlayerPrefs::shouldStartupWithVulkanRenderer()) ? RenderPath::Main : RenderPath::Psx;
    gNextRenderPath = gCurRenderPath;
    gbInitialized = true;
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tears down the renderer (the GL context must still be current)
//------------------------------------------------------------------------------------------------------------------------------------------
void destroy() noexcept {
    if (!gbInitialized)
        return;

    if (gbIsRendering) {
        endFrame();
    }

    destroyGlResources(false);
    GShaderCache::shutdown();       // Finishes the shader cache disk writes still queued
    VDrawing::shutdown();
    VRenderer::updateCoordSysInfo(0, 0);
    gbInitialized = false;
    gbIsRendering = false;
    gpWindow = nullptr;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Called when the GL context was lost and SDL made a new one (Android, 'SDL_RENDER_DEVICE_RESET'): all GL objects are gone, so make
// them all again. Everything they hold is in CPU memory (PSX VRAM etc.), and the shader programs come from the shader cache.
//------------------------------------------------------------------------------------------------------------------------------------------
void onContextLost() noexcept {
    if (!gbInitialized)
        return;

    SDL_Log("PsyDoom GLES: the GL context was lost, remaking all GL objects");
    destroyGlResources(true);
    initGlResources();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Android: no drawing while the app is in the background (there is no window surface then)
//------------------------------------------------------------------------------------------------------------------------------------------
void onAppEnterBackground() noexcept {
    gbAppInBackground = true;
}

void onAppEnterForeground() noexcept {
    gbAppInBackground = false;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Starts a frame. Returns 'true' if drawing can take place, 'isRendering()' tells the same afterwards.
//------------------------------------------------------------------------------------------------------------------------------------------
bool beginFrame() noexcept {
    // A frame begun and never ended: drop what was recorded for it
    if (gbIsRendering) {
        if (gCurRenderPath == RenderPath::Main) {
            VDrawing::endFrameGles();
        }

        gbIsRendering = false;
    }

    gbUsingGammaAdjustThisFrame = PlayerPrefs::isUsingGammaAdjust();
    gCurRenderPath = gNextRenderPath;

    if ((!gbInitialized) || gbAppInBackground)
        return false;

    // Follow the size of the window surface
    int drawableW = 0;
    int drawableH = 0;
    SDL_GL_GetDrawableSize(gpWindow, &drawableW, &drawableH);

    if ((drawableW <= 0) || (drawableH <= 0))
        return false;

    if (((uint32_t) drawableW != gScreenW) || ((uint32_t) drawableH != gScreenH)) {
        gScreenW = (uint32_t) drawableW;
        gScreenH = (uint32_t) drawableH;
        VRenderer::updateCoordSysInfo(gScreenW, gScreenH);
    }

    if (!ensureFrameHistory())
        return false;

    // Main render path: the frame is recorded by 'VDrawing' and drawn at the end of the frame.
    // Other render paths (except PSX, also drawn at the end) draw to the screen as they go.
    switch (gCurRenderPath) {
        case RenderPath::Main:
            VDrawing::beginFrameGles();
            break;

        case RenderPath::Crossfade:
        case RenderPath::LoadingPlaque:
        case RenderPath::Blit:
            beginScreenPass();
            break;

        case RenderPath::Psx:
            break;
    }

    gbIsRendering = true;
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tells if drawing can take place: if 'false' then NO draw commands should be issued
//------------------------------------------------------------------------------------------------------------------------------------------
bool isRendering() noexcept {
    return gbIsRendering;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Ends the frame and shows it on the screen (unless skipped)
//------------------------------------------------------------------------------------------------------------------------------------------
void endFrame() noexcept {
    if (!gbIsRendering)
        return;

    gbIsRendering = false;
    const bool bSkipPresent = (gbSkipNextFrame || gbSkipNextFramePresent);

    switch (gCurRenderPath) {
        case RenderPath::Main:  endFrame_Main(bSkipPresent);    break;
        case RenderPath::Psx:   endFrame_Psx(bSkipPresent);     break;
        default:                                                break;  // Already drawn to the screen
    }

    if (!bSkipPresent) {
        SDL_GL_SwapWindow(gpWindow);
    }

    gbSkipNextFrame = false;
    gbSkipNextFramePresent = false;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Render paths
//------------------------------------------------------------------------------------------------------------------------------------------
RenderPath getActiveRenderPath() noexcept { return gCurRenderPath; }
RenderPath getNextRenderPath() noexcept { return gNextRenderPath; }
void setNextRenderPath(const RenderPath renderPath) noexcept { gNextRenderPath = renderPath; }
bool isUsingPsxRenderPath() noexcept { return (gCurRenderPath == RenderPath::Psx); }
bool isNextRenderPathPsx() noexcept { return (gNextRenderPath == RenderPath::Psx); }
void switchToPsxRenderPath() noexcept { gNextRenderPath = RenderPath::Psx; }
void switchToMainRenderPath() noexcept { gNextRenderPath = RenderPath::Main; }

//------------------------------------------------------------------------------------------------------------------------------------------
// Notes that an area of PSX VRAM (of at least 1x1 pixels, can wrap around) was updated.
// The updates are applied at the end of the frame, before anything is drawn (as the Vulkan renderer does): straight from PSX VRAM.
//------------------------------------------------------------------------------------------------------------------------------------------
void pushPsxVramUpdates(const uint16_t rectLx, const uint16_t rectRx, const uint16_t rectTy, const uint16_t rectBy) noexcept {
    const Gpu::Core& psxGpu = PsxVm::gGpu;
    ASSERT(rectLx < psxGpu.ramPixelW);
    ASSERT(rectRx < psxGpu.ramPixelW);
    ASSERT(rectTy < psxGpu.ramPixelH);
    ASSERT(rectBy < psxGpu.ramPixelH);

    // Split up areas wrapping around VRAM
    if (rectRx < rectLx) {
        pushPsxVramUpdates(0, rectRx, rectTy, rectBy);
        pushPsxVramUpdates(rectLx, (uint16_t)(psxGpu.ramPixelW - 1), rectTy, rectBy);
        return;
    }

    if (rectBy < rectTy) {
        pushPsxVramUpdates(rectLx, rectRx, 0, rectBy);
        pushPsxVramUpdates(rectLx, rectRx, rectTy, (uint16_t)(psxGpu.ramPixelH - 1));
        return;
    }

    if (gbVramFullUploadPending)
        return;

    if (gPendingVramRects.size() >= MAX_PENDING_VRAM_RECTS) {
        gbVramFullUploadPending = true;
        gPendingVramRects.clear();
        return;
    }

    gPendingVramRects.push_back(VramRect{
        rectLx,
        rectTy,
        (uint16_t)(rectRx + 1 - rectLx),
        (uint16_t)(rectBy + 1 - rectTy)
    });
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Rebuilds the gamma adjustment remap table from the current gamma setting
//------------------------------------------------------------------------------------------------------------------------------------------
void rebuildGammaAdjustTex() noexcept {
    if (!gGammaLutTex) {
        gGammaLutTex = createTexture2d(GL_R8, 256, 1);
    }

    bindTextureForUpload(gGammaLutTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL_RED, GL_UNSIGNED_BYTE, Video::gGammaTbl.remapTbl8);

    // Stays bound to its texture unit for good
    glActiveTexture(GL_TEXTURE0 + GShaders::TEX_UNIT_GAMMA_LUT);
    glBindTexture(GL_TEXTURE_2D, gGammaLutTex);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Info and helpers for the crossfader, the loading plaque drawer and the display of external surfaces
//------------------------------------------------------------------------------------------------------------------------------------------
uint32_t getScreenWidth() noexcept { return gScreenW; }
uint32_t getScreenHeight() noexcept { return gScreenH; }
uint32_t getFrameHistoryIdx() noexcept { return gHistoryIdx; }

GLuint getFrameHistoryTex(const uint32_t idx) noexcept {
    return gHistoryTex[idx % NUM_FRAME_HISTORY];
}

bool didRenderToAllFrameHistory() noexcept {
    for (const bool bRendered : gbRenderedHistory) {
        if (!bRendered)
            return false;
    }

    return true;
}

END_NAMESPACE(GRenderer)

#endif  // #if PSYDOOM_GLES_RENDERER
