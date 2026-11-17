//------------------------------------------------------------------------------------------------------------------------------------------
// The OpenGL ES renderer, core module: frame lifecycle, render paths, framebuffers, PSX VRAM and the replay of what 'VDrawing'
// recorded for the frame, for both the OpenGL ES 3.0 and 2.0 paths. Written for tile based GPUs, see the header for the details.
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

// How MSAA is done for the main render path. The candidates are checked in this order (see 'getMsaaModeCandidates'): the tile based
// ones first, then the multisampled renderbuffer ones, of which only one is available on a given driver in practice.
enum class MsaaMode : uint8_t {
    None,
    RenderToTextureExt,     // EXT_multisampled_render_to_texture (OpenGL ES 2.0 and 3.0): samples only in tile memory, resolved as the tiles
                            // are written out. ANGLE (Vulkan) has it too: a transient multisampled image resolved by the render pass itself.
    RenderToTextureImg,     // IMG_multisampled_render_to_texture: the same (PowerVR)
    Renderbuffer,           // OpenGL ES 3.0: multisampled renderbuffer resolved by glBlitFramebuffer, then invalidated
    RenderbufferAngle,      // ANGLE_framebuffer_multisample + ANGLE_framebuffer_blit: the same with ANGLE's extensions (OpenGL ES 2.0)
    RenderbufferNv,         // NV_framebuffer_multisample + NV_framebuffer_blit: the same with NVIDIA's extensions (Tegra, OpenGL ES 2.0)
    RenderbufferApple       // APPLE_framebuffer_multisample: the same, resolved with glResolveMultisampleFramebufferAPPLE
};

static constexpr bool isRenderbufferMsaa(const MsaaMode msaaMode) noexcept {
    return (msaaMode >= MsaaMode::Renderbuffer);
}

// A texture format. OpenGL ES 3.0 makes textures with 'glTexStorage2D' (sized internal format), OpenGL ES 2.0 with 'glTexImage2D',
// where the internal format is unsized (the same as 'format') and the format and type decide the storage.
struct TexFormat {
    GLenum  internalFormat;
    GLenum  format;             // Format and type of the pixels uploaded
    GLenum  type;
    GLenum  renderbufferFormat; // The matching sized format for a multisampled renderbuffer resolved into the texture, '0' if none
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

// Uniform block of the draw programs, std140 layout ('VShaderUniforms_Draw' + padding to the block size). OpenGL ES 3.0 only.
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
    GLuint      ubo;        // OpenGL ES 3.0 only
    GLuint      vao;        // OpenGL ES 3.0 only
    GLsizeiptr  vboSize;
    GLsizeiptr  uboSize;
    GLsync      fence;      // OpenGL ES 3.0 only: signalled once the GPU is done with the frame that used the slot
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

// What the device and the API path can do
struct Caps {
    bool    bAngle;                 // The driver is ANGLE (OpenGL ES on top of Vulkan, Direct3D etc.)
    bool    bUnpackSubimage;        // GL_UNPACK_ROW_LENGTH and the skips: OpenGL ES 3.0 or EXT_unpack_subimage
    bool    bDiscardFramebuffer;    // OpenGL ES 2.0: EXT_discard_framebuffer (OpenGL ES 3.0 has glInvalidateFramebuffer)
    bool    bHighpFragment;         // 'highp' floats in fragment shaders (always with OpenGL ES 3.0)
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
static GApi         gApi = GApi::Gles3;
static Caps         gCaps;
static GLint        gMaxTextureSize;
static GLint        gUboOffsetAlignment;
static MsaaMode     gMsaaMode;
static GLsizei      gNumSamples;            // Samples of the main render path, '1' without MSAA
static bool         gbPerSampleShading;     // Sample rate shading for the 3D world when MSAA is on (OpenGL ES 3.0: OES_sample_shading)
static TexFormat    gDrawColorFormat;       // Color format of the frames drawn: 16-bit like the PSX, or 32-bit
static TexFormat    gVramFormat;            // Format of PSX VRAM, the staging textures and the PSX framebuffer: 16 bits per texel
static TexFormat    gGammaLutFormat;        // Format of the gamma remap table: 8 bits per texel
static TexFormat    gRgba8Format;           // RGBA8 textures (loading plaques, movies and logos)
static bool         gbStagedVramUpdates;    // Whether PSX VRAM updates are copied in by the GPU
static bool         gbInstancingAllowed = true;     // Whether instancing may be used (Android: set by the launcher through JNA)
static bool         gbInstancing;           // OpenGL ES 3.0: each primitive of the main render path is drawn as an instance ('VInstance_Draw')
static bool         gbBaseInstance;         // Instancing: draws start at their first instance with EXT_base_instance (else attributes move)

// Size of the screen (window surface) in pixels
static uint32_t     gScreenW;
static uint32_t     gScreenH;

// Main render path: the frames drawn (the latest ones are kept for crossfades and loading plaques).
// With multisampled render to texture the framebuffers render multisampled to the single sample textures directly.
static GLuint       gHistoryTex[NUM_FRAME_HISTORY];
static GLuint       gHistoryFbo[NUM_FRAME_HISTORY];
static bool         gbRenderedHistory[NUM_FRAME_HISTORY];
static uint32_t     gHistoryIdx;                            // Which frame the next frame of the main render path is drawn to
static uint32_t     gHistoryW;
static uint32_t     gHistoryH;
static GLuint       gMsaaRenderbuffer;                      // MSAA with a renderbuffer only
static GLuint       gMsaaFbo;

// PSX render path: the PSX framebuffer, as is
static GLuint       gPsxFbTex[NUM_PSX_FB_TEXTURES];
static uint32_t     gPsxFbTexIdx;

// PSX VRAM, its pending updates and the staging textures used to copy them in
static GLuint                   gVramTex;
static GLuint                   gVramFbo;                   // OpenGL ES 3.0 only
static GLuint                   gStagingTex[NUM_STAGING_TEX];
static GLuint                   gStagingFbo[NUM_STAGING_TEX];
static uint32_t                 gStagingIdx;
static std::vector<VramRect>    gPendingVramRects;
static bool                     gbVramFullUploadPending;
static std::vector<VramBlit>    gVramBlits;
static std::vector<uint16_t>    gVramUploadRows;            // Without GL_UNPACK_ROW_LENGTH: the rows of an area uploaded, gathered

// Gamma adjustment remap table (256x1)
static GLuint gGammaLutTex;

// Samplers for the post programs (textures sampled with normalized coordinates). OpenGL ES 3.0 only: the filter is set on the
// texture itself with OpenGL ES 2.0.
static GLuint gSamplerNearest;
static GLuint gSamplerLinear;

// Streaming buffers for the frames of the main render path, and the corners of the quads drawn by the post programs
static StreamSlot   gStreamSlots[NUM_STREAM_SLOTS];
static uint32_t     gStreamSlotIdx;
static GLuint       gQuadVbo;
static GLuint       gQuadVao;               // OpenGL ES 3.0 only
static GLuint       gNumEnabledAttribs;     // OpenGL ES 2.0 only (no VAOs): how many vertex attributes are enabled (from 0)

// Uniform data for the current frame, as uploaded (OpenGL ES 3.0)
static std::vector<std::byte> gUniformUploadBuffer;

// OpenGL ES 2.0 (no uniform blocks): the draw uniforms in effect and which of them each draw program has.
// The serial number changes whenever the uniforms do: a program gets them before its next draw if its number is not the same.
static const VShaderUniforms_Draw   gZeroDrawUniforms = {};
static const VShaderUniforms_Draw*  gpDrawUniforms = &gZeroDrawUniforms;
static uint32_t                     gDrawUniformsSerial;
static uint32_t                     gProgramUniformsSerial[NUM_DRAW_PROGRAMS];

static StateCache gStateCache;

static bool isGles3() noexcept {
    return (gApi == GApi::Gles3);
}

// Clears the GL error flags (bounded: a lost context can keep reporting errors)
static void clearGlErrors() noexcept {
    for (uint32_t i = 0; (i < 32) && (glGetError() != GL_NO_ERROR); ++i) {}
}

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

// OpenGL ES 2.0 (no VAOs): enables vertex attributes 0 to 'numAttribs - 1' and disables the rest
static void setNumEnabledVertexAttribs(const GLuint numAttribs) noexcept {
    while (gNumEnabledAttribs < numAttribs) {
        glEnableVertexAttribArray(gNumEnabledAttribs++);
    }

    while (gNumEnabledAttribs > numAttribs) {
        glDisableVertexAttribArray(--gNumEnabledAttribs);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Waits until the GPU has signalled a fence and deletes it (OpenGL ES 3.0)
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
// Texture and framebuffer helpers
//------------------------------------------------------------------------------------------------------------------------------------------
void bindTextureForUpload(const GLuint texture) noexcept {
    glActiveTexture(GL_TEXTURE0 + TEX_UNIT_UPLOAD);
    glBindTexture(GL_TEXTURE_2D, texture);
}

static GLuint createTexture2d(const TexFormat& format, const GLsizei width, const GLsizei height) noexcept {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    bindTextureForUpload(texture);

    if (isGles3()) {
        glTexStorage2D(GL_TEXTURE_2D, 1, format.internalFormat, width, height);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, (GLint) format.internalFormat, width, height, 0, format.format, format.type, nullptr);
    }

    // Note: integer textures are only complete with nearest filtering, and OpenGL ES 2.0 only allows clamping for non power of two sizes
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

GLuint createRgba8Texture(const uint32_t width, const uint32_t height) noexcept {
    return createTexture2d(gRgba8Format, (GLsizei) width, (GLsizei) height);
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

static void deleteBuffer(GLuint& buffer, const bool bContextLost) noexcept {
    if (buffer && (!bContextLost)) {
        glDeleteBuffers(1, &buffer);
    }

    buffer = 0;
}

static void deleteVertexArray(GLuint& vao, const bool bContextLost) noexcept {
    if (vao && (!bContextLost)) {
        glDeleteVertexArrays(1, &vao);
    }

    vao = 0;
}

// Makes a framebuffer drawing to the given texture, with the given MSAA mode and sample count (render to texture modes only)
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
// MSAA with a multisampled renderbuffer: storage, resolve and discard for each of the ways it can be done.
// Note: the read and draw framebuffer binding points of ANGLE_framebuffer_blit, NV_framebuffer_blit and APPLE_framebuffer_multisample
// have the same enums as those of OpenGL ES 3.0.
//------------------------------------------------------------------------------------------------------------------------------------------
static void renderbufferStorageMultisample(
    const MsaaMode msaaMode,
    const GLsizei numSamples,
    const GLenum format,
    const GLsizei width,
    const GLsizei height
) noexcept {
    switch (msaaMode) {
        case MsaaMode::Renderbuffer:        glRenderbufferStorageMultisample(GL_RENDERBUFFER, numSamples, format, width, height);       break;
        case MsaaMode::RenderbufferAngle:   glRenderbufferStorageMultisampleANGLE(GL_RENDERBUFFER, numSamples, format, width, height);  break;
        case MsaaMode::RenderbufferNv:      glRenderbufferStorageMultisampleNV(GL_RENDERBUFFER, numSamples, format, width, height);     break;
        case MsaaMode::RenderbufferApple:   glRenderbufferStorageMultisampleAPPLE(GL_RENDERBUFFER, numSamples, format, width, height);  break;
        default: break;
    }
}

// Resolves all of the samples of one framebuffer into another of the same size.
// ANGLE (Vulkan) turns a resolve done right after drawing, covering the whole framebuffer and with no scissor, into a resolve attachment
// of the render pass: the samples then stay in tile memory, as with multisampled render to texture.
static void resolveMultisample(const MsaaMode msaaMode, const GLuint srcFbo, const GLuint dstFbo, const GLint width, const GLint height) noexcept {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, srcFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstFbo);

    switch (msaaMode) {
        case MsaaMode::Renderbuffer:
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            break;

        case MsaaMode::RenderbufferAngle:
            glBlitFramebufferANGLE(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            break;

        case MsaaMode::RenderbufferNv:
            glBlitFramebufferNV(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            break;

        case MsaaMode::RenderbufferApple:
            glResolveMultisampleFramebufferAPPLE();
            break;

        default:
            break;
    }
}

// Tells the driver the contents of the color attachment of a framebuffer are not needed anymore, so they are never written to memory
static void discardColorAttachment(const GLuint fbo) noexcept {
    const GLenum attachment = GL_COLOR_ATTACHMENT0;

    if (isGles3()) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glInvalidateFramebuffer(GL_READ_FRAMEBUFFER, 1, &attachment);
    } else if (gCaps.bDiscardFramebuffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glDiscardFramebufferEXT(GL_FRAMEBUFFER, 1, &attachment);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Device capabilities: decides the texture formats, the color format of the frames drawn and how MSAA is done
//------------------------------------------------------------------------------------------------------------------------------------------

// Tells if frames can be drawn with the given format and MSAA, by making a small framebuffer (and resolving it, for renderbuffers)
static bool canDrawWith(const TexFormat& colorFormat, const MsaaMode msaaMode, const GLsizei numSamples) noexcept {
    constexpr GLsizei TEST_SIZE = 64;

    if (isRenderbufferMsaa(msaaMode) && (!colorFormat.renderbufferFormat))
        return false;

    clearGlErrors();
    GLuint texture = createTexture2d(colorFormat, TEST_SIZE, TEST_SIZE);
    GLuint fbo = 0;
    GLuint msaaFbo = 0;
    GLuint renderbuffer = 0;
    bool bOk = false;

    if (isRenderbufferMsaa(msaaMode)) {
        glGenRenderbuffers(1, &renderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
        renderbufferStorageMultisample(msaaMode, numSamples, colorFormat.renderbufferFormat, TEST_SIZE, TEST_SIZE);
        glGenFramebuffers(1, &msaaFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, msaaFbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
        bOk = isBoundFramebufferComplete();

        // The texture the samples are resolved to must also be a valid color attachment, and the resolve must be allowed
        // (the formats must match, which completeness doesn't tell)
        if (bOk) {
            fbo = createTextureFbo(texture, MsaaMode::None, 1);
            bOk = isBoundFramebufferComplete();
        }

        if (bOk) {
            glBindFramebuffer(GL_FRAMEBUFFER, msaaFbo);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            resolveMultisample(msaaMode, msaaFbo, fbo, TEST_SIZE, TEST_SIZE);
            bOk = (glGetError() == GL_NO_ERROR);
        }
    } else {
        fbo = createTextureFbo(texture, msaaMode, numSamples);
        bOk = isBoundFramebufferComplete() && (glGetError() == GL_NO_ERROR);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    deleteFramebuffer(fbo, false);
    deleteFramebuffer(msaaFbo, false);
    deleteRenderbuffer(renderbuffer, false);
    deleteTexture(texture, false);
    clearGlErrors();    // Formats or sample counts that are not supported
    return bOk;
}

// OpenGL ES 3.0: the most samples a format supports for a multisampled renderbuffer
static GLsizei getMaxRenderbufferSamples(const GLenum format) noexcept {
    GLint numSampleCounts = 0;
    glGetInternalformativ(GL_RENDERBUFFER, format, GL_NUM_SAMPLE_COUNTS, 1, &numSampleCounts);

    if (numSampleCounts <= 0)
        return 1;

    std::vector<GLint> sampleCounts((size_t) numSampleCounts);
    glGetInternalformativ(GL_RENDERBUFFER, format, GL_SAMPLES, numSampleCounts, sampleCounts.data());
    return std::max(*std::max_element(sampleCounts.begin(), sampleCounts.end()), 1);   // Note: listed in descending order, but be safe
}

// The most samples an MSAA mode can do with a format
static GLsizei getMaxSamples(const MsaaMode msaaMode, const TexFormat& colorFormat) noexcept {
    // Note: GL_MAX_SAMPLES_EXT, GL_MAX_SAMPLES_ANGLE, GL_MAX_SAMPLES_NV and GL_MAX_SAMPLES_APPLE are the same enum as GL_MAX_SAMPLES
    GLint maxSamples = 0;
    glGetIntegerv((msaaMode == MsaaMode::RenderToTextureImg) ? GL_MAX_SAMPLES_IMG : GL_MAX_SAMPLES, &maxSamples);

    if ((msaaMode == MsaaMode::Renderbuffer) && colorFormat.renderbufferFormat) {
        maxSamples = std::min(maxSamples, getMaxRenderbufferSamples(colorFormat.renderbufferFormat));
    }

    clearGlErrors();
    return std::max<GLsizei>(maxSamples, 1);
}

// The ways MSAA can be done on this driver, best first: everything is checked through the extensions (OpenGL ES 2.0 has no MSAA of its
// own, ANGLE has its own extensions for it), plus the multisampled renderbuffers of OpenGL ES 3.0.
static std::vector<MsaaMode> getMsaaModeCandidates() noexcept {
    std::vector<MsaaMode> modes;

    if (GLAD_GL_EXT_multisampled_render_to_texture && glFramebufferTexture2DMultisampleEXT) {
        modes.push_back(MsaaMode::RenderToTextureExt);
    }

    if (GLAD_GL_IMG_multisampled_render_to_texture && glFramebufferTexture2DMultisampleIMG) {
        modes.push_back(MsaaMode::RenderToTextureImg);
    }

    if (isGles3()) {
        modes.push_back(MsaaMode::Renderbuffer);
    }

    if (GLAD_GL_ANGLE_framebuffer_multisample && GLAD_GL_ANGLE_framebuffer_blit && glRenderbufferStorageMultisampleANGLE && glBlitFramebufferANGLE) {
        modes.push_back(MsaaMode::RenderbufferAngle);
    }

    if (GLAD_GL_NV_framebuffer_multisample && GLAD_GL_NV_framebuffer_blit && glRenderbufferStorageMultisampleNV && glBlitFramebufferNV) {
        modes.push_back(MsaaMode::RenderbufferNv);
    }

    if (GLAD_GL_APPLE_framebuffer_multisample && glRenderbufferStorageMultisampleAPPLE && glResolveMultisampleFramebufferAPPLE) {
        modes.push_back(MsaaMode::RenderbufferApple);
    }

    return modes;
}

static const char* getMsaaModeName(const MsaaMode msaaMode) noexcept {
    switch (msaaMode) {
        case MsaaMode::RenderToTextureExt:  return "EXT_multisampled_render_to_texture";
        case MsaaMode::RenderToTextureImg:  return "IMG_multisampled_render_to_texture";
        case MsaaMode::Renderbuffer:        return "renderbuffer + blit";
        case MsaaMode::RenderbufferAngle:   return "ANGLE_framebuffer_multisample + ANGLE_framebuffer_blit";
        case MsaaMode::RenderbufferNv:      return "NV_framebuffer_multisample + NV_framebuffer_blit";
        case MsaaMode::RenderbufferApple:   return "APPLE_framebuffer_multisample";
        default:                            return "off";
    }
}

static void decideDrawFormatAndMsaa() noexcept {
    // Formats to try: 16-bit color like the Vulkan renderer (the PSX's precision when blending) unless 32-bit shading is wanted.
    // Dithering is turned off (see 'initGlState'): it would change the 16-bit color, which Vulkan does not do.
    std::vector<TexFormat> colorFormats;

    if (isGles3()) {
        if (!Config::gbUseVulkan32BitShading) {
            colorFormats.push_back({ GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, GL_RGB5_A1 });
        }

        colorFormats.push_back({ GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_RGBA8 });
    } else {
        // OpenGL ES 2.0: RGBA8 renderbuffers need OES_rgb8_rgba8 (or ARM_rgba8)
        if (!Config::gbUseVulkan32BitShading) {
            colorFormats.push_back({ GL_RGBA, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, GL_RGB5_A1 });
        }

        const bool bRgba8Renderbuffers = (GLAD_GL_OES_rgb8_rgba8 || GLAD_GL_ARM_rgba8);
        colorFormats.push_back({ GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, (GLenum)((bRgba8Renderbuffers) ? GL_RGBA8_OES : 0) });
    }

    // MSAA: try each format, then each MSAA mode, with the most samples wanted and possible first, then fewer.
    // The first combination that works is used: MSAA is preferred over 16-bit color, the tile based MSAA modes over the others.
    const int32_t wantedSamples = std::max(Config::gAAMultisamples, 1);

    if (wantedSamples > 1) {
        const std::vector<MsaaMode> msaaModes = getMsaaModeCandidates();

        for (const TexFormat& colorFormat : colorFormats) {
            for (const MsaaMode msaaMode : msaaModes) {
                const GLsizei maxSamples = getMaxSamples(msaaMode, colorFormat);
                GLsizei numSamples = 1;

                while ((numSamples * 2 <= wantedSamples) && (numSamples * 2 <= maxSamples)) {
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
        }
    }

    // No MSAA (or not possible): the first format that works
    for (const TexFormat& colorFormat : colorFormats) {
        if (canDrawWith(colorFormat, MsaaMode::None, 1)) {
            gDrawColorFormat = colorFormat;
            gMsaaMode = MsaaMode::None;
            gNumSamples = 1;
            return;
        }
    }

    FatalErrors::raise("GLES renderer: no color format can be drawn to!");
}

// Decides the formats of PSX VRAM etc.: R16UI with OpenGL ES 3.0. OpenGL ES 2.0 has no integer textures: 2 bytes per texel, RG8 if
// there is EXT_texture_rg (renderable, so VRAM updates can be copied in by the GPU), otherwise luminance alpha.
static void decideTextureFormats() noexcept {
    if (isGles3()) {
        gVramFormat = { GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, 0 };
        gGammaLutFormat = { GL_R8, GL_RED, GL_UNSIGNED_BYTE, 0 };
        gRgba8Format = { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 0 };
        return;
    }

    gVramFormat = { GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, 0 };
    gGammaLutFormat = { GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0 };
    gRgba8Format = { GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, 0 };

    if (GLAD_GL_EXT_texture_rg) {
        const TexFormat rg8Format = { GL_RG_EXT, GL_RG_EXT, GL_UNSIGNED_BYTE, 0 };
        clearGlErrors();
        GLuint texture = createTexture2d(rg8Format, 4, 4);
        const bool bOk = (glGetError() == GL_NO_ERROR);
        deleteTexture(texture, false);
        clearGlErrors();

        if (bOk) {
            gVramFormat = rg8Format;
        }
    }
}

// Tells if fragment shaders have 'highp' floats (always with OpenGL ES 3.0)
static bool hasHighpFragmentFloats() noexcept {
    if (isGles3())
        return true;

    GLint range[2] = {};
    GLint precision = 0;
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_HIGH_FLOAT, range, &precision);
    clearGlErrors();
    return (precision > 0);
}

// Gets what the device can do: returns 'false' if the renderer can't run on it
static bool determineCaps() noexcept {
    const char* const pVendor = (const char*) glGetString(GL_VENDOR);
    const char* const pRenderer = (const char*) glGetString(GL_RENDERER);
    const char* const pVersion = (const char*) glGetString(GL_VERSION);

    gCaps = {};
    gCaps.bAngle = (pRenderer && std::strstr(pRenderer, "ANGLE")) || (pVersion && std::strstr(pVersion, "ANGLE"));
    gCaps.bUnpackSubimage = (isGles3() || GLAD_GL_EXT_unpack_subimage);
    gCaps.bDiscardFramebuffer = ((!isGles3()) && GLAD_GL_EXT_discard_framebuffer && glDiscardFramebufferEXT);
    gCaps.bHighpFragment = hasHighpFragmentFloats();

    // OpenGL ES 2.0: the draw programs decode PSX VRAM with float arithmetic on coordinates up to 8192, which needs 'highp' floats
    if (!gCaps.bHighpFragment) {
        SDL_Log("PsyDoom GLES: no 'highp' floats in fragment shaders, the OpenGL ES 2.0 renderer can't be used on '%s'", pRenderer);
        return false;
    }

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gMaxTextureSize);
    gUboOffsetAlignment = 1;

    if (isGles3()) {
        glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &gUboOffsetAlignment);
        gUboOffsetAlignment = std::max(gUboOffsetAlignment, 1);
    }

    decideTextureFormats();
    gDrawColorFormat = {};
    gMsaaMode = MsaaMode::None;
    gNumSamples = 1;
    decideDrawFormatAndMsaa();

    // Sample rate shading as with Vulkan: the textures antialiased too, unless only the edges are wanted ('AntiAliasingEdgesOnly').
    // MSAA itself is nearly free on tile based GPUs, sample rate shading multiplies the fragment shading work by the sample count.
    // OES_sample_shading needs OpenGL ES 3.0.
    gbPerSampleShading = (
        isGles3() &&
        (gNumSamples > 1) &&
        (!Config::gbAAEdgesOnly) &&
        GLAD_GL_OES_sample_shading &&
        (glMinSampleShadingOES != nullptr)
    );

    // Instancing: core in OpenGL ES 3.0 (the OpenGL ES 2.0 path draws vertices).
    // Draws start at their first instance with EXT_base_instance, otherwise the instance attributes are moved to it for each draw.
    gbInstancing = (
        isGles3() &&
        gbInstancingAllowed &&
        (glDrawArraysInstanced != nullptr) &&
        (glVertexAttribDivisor != nullptr)
    );

    gbBaseInstance = (gbInstancing && GLAD_GL_EXT_base_instance && (glDrawArraysInstancedBaseInstanceEXT != nullptr));

    const bool b16BitColor = (gDrawColorFormat.type == GL_UNSIGNED_SHORT_5_5_5_1);
    const char* const instancingStr = (gbBaseInstance) ? " | instancing (base instance)" : (gbInstancing) ? " | instancing" : "";

    SDL_Log(
        "PsyDoom GLES: %s | %s | %s | OpenGL ES %s path%s | color %s | MSAA %dx (%s, wanted %dx)%s%s",
        pVendor,
        pRenderer,
        pVersion,
        (isGles3()) ? "3.0" : "2.0",
        (gCaps.bAngle) ? " (ANGLE)" : "",
        (b16BitColor) ? "16-bit" : "32-bit",
        (int) gNumSamples,
        getMsaaModeName(gMsaaMode),
        (int) std::max(Config::gAAMultisamples, 1),
        (gbPerSampleShading) ? ", sample rate shading" : "",
        instancingStr
    );

    return true;
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

    if (isRenderbufferMsaa(gMsaaMode)) {
        glGenRenderbuffers(1, &gMsaaRenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, gMsaaRenderbuffer);
        renderbufferStorageMultisample(gMsaaMode, gNumSamples, gDrawColorFormat.renderbufferFormat, (GLsizei) fbW, (GLsizei) fbH);
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

// Uploads an area of PSX VRAM straight from the PSX GPU's memory to a texture of the VRAM format.
// Without GL_UNPACK_ROW_LENGTH (OpenGL ES 2.0 without EXT_unpack_subimage) the rows are gathered first, unless they are whole rows.
static void uploadVramRectDirect(const GLuint texture, const GLint dstX, const GLint dstY, const VramRect& rect) noexcept {
    const Gpu::Core& psxGpu = PsxVm::gGpu;
    bindTextureForUpload(texture);

    if (rect.w == psxGpu.ramPixelW) {
        const uint16_t* const pSrc = psxGpu.pRam + (size_t) rect.y * psxGpu.ramPixelW;
        glTexSubImage2D(GL_TEXTURE_2D, 0, dstX, dstY, rect.w, rect.h, gVramFormat.format, gVramFormat.type, pSrc);
    } else if (gCaps.bUnpackSubimage) {
        // Note: the EXT_unpack_subimage enums are the same as those of OpenGL ES 3.0
        glPixelStorei(GL_UNPACK_ROW_LENGTH, psxGpu.ramPixelW);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, rect.x);
        glPixelStorei(GL_UNPACK_SKIP_ROWS, rect.y);
        glTexSubImage2D(GL_TEXTURE_2D, 0, dstX, dstY, rect.w, rect.h, gVramFormat.format, gVramFormat.type, psxGpu.pRam);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    } else {
        gVramUploadRows.resize((size_t) rect.w * rect.h);

        for (uint32_t y = 0; y < rect.h; ++y) {
            std::memcpy(
                gVramUploadRows.data() + (size_t) y * rect.w,
                psxGpu.pRam + (size_t)(rect.y + y) * psxGpu.ramPixelW + rect.x,
                (size_t) rect.w * sizeof(uint16_t)
            );
        }

        glTexSubImage2D(GL_TEXTURE_2D, 0, dstX, dstY, rect.w, rect.h, gVramFormat.format, gVramFormat.type, gVramUploadRows.data());
    }
}

// Copies an area of a staging texture into VRAM on the GPU (the staging texture's framebuffer is bound for reading)
static void copyStagingToVram(const VramBlit& blit) noexcept {
    if (isGles3()) {
        glBlitFramebuffer(
            blit.srcX, blit.srcY, blit.srcX + blit.w, blit.srcY + blit.h,
            blit.dstX, blit.dstY, blit.dstX + blit.w, blit.dstY + blit.h,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST
        );
    } else {
        // OpenGL ES 2.0: no blits, but a copy from the framebuffer bound into a texture
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, blit.dstX, blit.dstY, blit.srcX, blit.srcY, blit.w, blit.h);
    }
}

// Binds a staging texture's framebuffer for reading and VRAM as where copies go
static void bindStagingForCopy(const uint32_t stagingIdx) noexcept {
    if (isGles3()) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gStagingFbo[stagingIdx]);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gVramFbo);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, gStagingFbo[stagingIdx]);
        bindTextureForUpload(gVramTex);
    }
}

// Sets up the staging textures (VRAM must exist): returns 'false' if VRAM updates can't be copied in by the GPU.
//  - OpenGL ES 3.0: VRAM (R16UI) is a color attachment, copies are blits.
//  - OpenGL ES 2.0: copies are 'glCopyTexSubImage2D' from the staging texture's framebuffer, so the format must be renderable (RG8).
static bool initStagingTextures() noexcept {
    if (isGles3()) {
        glGenFramebuffers(1, &gVramFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, gVramFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gVramTex, 0);

        if (!isBoundFramebufferComplete())
            return false;
    } else if (gVramFormat.format != GL_RG_EXT) {
        return false;
    }

    for (uint32_t i = 0; i < NUM_STAGING_TEX; ++i) {
        gStagingTex[i] = createTexture2d(gVramFormat, STAGING_TEX_W, STAGING_TEX_H);
        gStagingFbo[i] = createTextureFbo(gStagingTex[i], MsaaMode::None, 1);

        if (!isBoundFramebufferComplete())
            return false;
    }

    // OpenGL ES 2.0: make sure a copy works, before VRAM gets its contents
    if (!isGles3()) {
        clearGlErrors();
        bindStagingForCopy(0);
        copyStagingToVram(VramBlit{ 0, 0, 0, 0, 1, 1 });

        if (glGetError() != GL_NO_ERROR)
            return false;
    }

    return true;
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
    gVramUploadRows.clear();
    gVramUploadRows.shrink_to_fit();
    gbVramFullUploadPending = false;
    gbStagedVramUpdates = false;
}

static void initVram() noexcept {
    Gpu::Core& psxGpu = PsxVm::gGpu;
    gVramTex = createTexture2d(gVramFormat, psxGpu.ramPixelW, psxGpu.ramPixelH);
    gPendingVramRects.clear();
    gbVramFullUploadPending = false;
    gStagingIdx = 0;

    // The GPU copies VRAM updates in from the staging textures where possible
    gbStagedVramUpdates = initStagingTextures();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    clearGlErrors();

    if (!gbStagedVramUpdates) {
        SDL_Log("PsyDoom GLES: PSX VRAM can't be copied into by the GPU, updates to it are uploaded directly");

        for (uint32_t i = 0; i < NUM_STAGING_TEX; ++i) {
            deleteFramebuffer(gStagingFbo[i], false);
            deleteTexture(gStagingTex[i], false);
        }

        deleteFramebuffer(gVramFbo, false);
    }

    uploadVramRectDirect(gVramTex, 0, 0, VramRect{ 0, 0, psxGpu.ramPixelW, psxGpu.ramPixelH });

    // OpenGL ES 2.0: the draw programs read VRAM with normalized texture coordinates, so they need its size
    if (!isGles3()) {
        GShaders::setVramSize(psxGpu.ramPixelW, psxGpu.ramPixelH);
        invalidateStateCache();
    }

    // VRAM stays bound to its texture unit for good
    glActiveTexture(GL_TEXTURE0 + GShaders::TEX_UNIT_VRAM);
    glBindTexture(GL_TEXTURE_2D, gVramTex);
}

// Copies the areas placed in the current staging texture into VRAM, on the GPU
static void flushVramBlits() noexcept {
    if (gVramBlits.empty())
        return;

    setScissorTest(false);
    bindStagingForCopy(gStagingIdx);

    for (const VramBlit& blit : gVramBlits) {
        copyStagingToVram(blit);
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
// Vertex buffers: streaming buffers (vertices and uniforms of the frames of the main render path) and the corners of the post quads
//------------------------------------------------------------------------------------------------------------------------------------------

// Points the draw vertex attributes at the bound vertex buffer: the 'VVertex_Draw' layout (see 'DECLARE_VS_INPUTS_VVERTEX_DRAW' in the
// Vulkan shaders). OpenGL ES 2.0 has no integer attributes: they become floats (not normalized) there.
static void setDrawVertexAttribPointers() noexcept {
    constexpr GLsizei STRIDE = (GLsizei) sizeof(VVertex_Draw);
    static_assert(sizeof(VVertex_Draw) == 40);

    glVertexAttribPointer(GShaders::ATTRIB_POS, 3, GL_FLOAT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, x));
    glVertexAttribPointer(GShaders::ATTRIB_UV, 2, GL_FLOAT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, u));

    if (isGles3()) {
        glVertexAttribIPointer(GShaders::ATTRIB_COLOR_LIGHTDIM, 4, GL_UNSIGNED_BYTE, STRIDE, (const void*) offsetof(VVertex_Draw, r));
        glVertexAttribIPointer(GShaders::ATTRIB_TEXWIN_POS, 2, GL_UNSIGNED_SHORT, STRIDE, (const void*) offsetof(VVertex_Draw, texWinX));
        glVertexAttribIPointer(GShaders::ATTRIB_TEXWIN_SIZE, 2, GL_UNSIGNED_SHORT, STRIDE, (const void*) offsetof(VVertex_Draw, texWinW));
        glVertexAttribIPointer(GShaders::ATTRIB_CLUT_POS, 2, GL_UNSIGNED_SHORT, STRIDE, (const void*) offsetof(VVertex_Draw, clutX));
        glVertexAttribIPointer(GShaders::ATTRIB_STMUL, 4, GL_UNSIGNED_BYTE, STRIDE, (const void*) offsetof(VVertex_Draw, stmulR));
    } else {
        glVertexAttribPointer(GShaders::ATTRIB_COLOR_LIGHTDIM, 4, GL_UNSIGNED_BYTE, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, r));
        glVertexAttribPointer(GShaders::ATTRIB_TEXWIN_POS, 2, GL_UNSIGNED_SHORT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, texWinX));
        glVertexAttribPointer(GShaders::ATTRIB_TEXWIN_SIZE, 2, GL_UNSIGNED_SHORT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, texWinW));
        glVertexAttribPointer(GShaders::ATTRIB_CLUT_POS, 2, GL_UNSIGNED_SHORT, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, clutX));
        glVertexAttribPointer(GShaders::ATTRIB_STMUL, 4, GL_UNSIGNED_BYTE, GL_FALSE, STRIDE, (const void*) offsetof(VVertex_Draw, stmulR));
    }
}

// Instancing (OpenGL ES 3.0): points the instance attributes at the bound buffer, starting at the given instance. The 'VInstance_Draw'
// layout: positions and texture coordinates as one vec4 per component (a value per corner), the corner colors as one uint per corner,
// the texture window as 4 ushorts and the CLUT position and semi-transparency multiply as 2 uints.
static void setDrawInstanceAttribPointers(const uint32_t firstInstance) noexcept {
    constexpr GLsizei STRIDE = (GLsizei) sizeof(VInstance_Draw);
    static_assert(offsetof(VInstance_Draw, stmulR) == offsetof(VInstance_Draw, clutX) + 4);

    const uintptr_t base = (uintptr_t) firstInstance * sizeof(VInstance_Draw);
    const auto offset = [=](const size_t fieldOffset) noexcept { return (const void*)(base + fieldOffset); };

    glVertexAttribPointer(GShaders::INST_ATTRIB_X, 4, GL_FLOAT, GL_FALSE, STRIDE, offset(offsetof(VInstance_Draw, x)));
    glVertexAttribPointer(GShaders::INST_ATTRIB_Y, 4, GL_FLOAT, GL_FALSE, STRIDE, offset(offsetof(VInstance_Draw, y)));
    glVertexAttribPointer(GShaders::INST_ATTRIB_Z, 4, GL_FLOAT, GL_FALSE, STRIDE, offset(offsetof(VInstance_Draw, z)));
    glVertexAttribPointer(GShaders::INST_ATTRIB_U, 4, GL_FLOAT, GL_FALSE, STRIDE, offset(offsetof(VInstance_Draw, u)));
    glVertexAttribPointer(GShaders::INST_ATTRIB_V, 4, GL_FLOAT, GL_FALSE, STRIDE, offset(offsetof(VInstance_Draw, v)));
    glVertexAttribIPointer(GShaders::INST_ATTRIB_COLORS, 4, GL_UNSIGNED_INT, STRIDE, offset(offsetof(VInstance_Draw, color)));
    glVertexAttribIPointer(GShaders::INST_ATTRIB_TEXWIN, 4, GL_UNSIGNED_SHORT, STRIDE, offset(offsetof(VInstance_Draw, texWinX)));
    glVertexAttribIPointer(GShaders::INST_ATTRIB_CLUT_STMUL, 2, GL_UNSIGNED_INT, STRIDE, offset(offsetof(VInstance_Draw, clutX)));
}

static void initStreamSlots() noexcept {
    for (StreamSlot& slot : gStreamSlots) {
        slot = {};
        glGenBuffers(1, &slot.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, slot.vbo);
        slot.vboSize = 1024 * 1024;
        glBufferData(GL_ARRAY_BUFFER, slot.vboSize, nullptr, GL_STREAM_DRAW);

        // OpenGL ES 3.0: a VAO per slot and the uniform buffer (OpenGL ES 2.0 sets the attributes and uniforms as it draws).
        // With instancing the attributes advance once per instance.
        if (isGles3()) {
            glGenVertexArrays(1, &slot.vao);
            glBindVertexArray(slot.vao);

            if (gbInstancing) {
                for (GLuint attrib = 0; attrib < GShaders::NUM_INST_ATTRIBS; ++attrib) {
                    glEnableVertexAttribArray(attrib);
                    glVertexAttribDivisor(attrib, 1);
                }

                setDrawInstanceAttribPointers(0);
            } else {
                for (GLuint attrib = 0; attrib < GShaders::NUM_DRAW_ATTRIBS; ++attrib) {
                    glEnableVertexAttribArray(attrib);
                }

                setDrawVertexAttribPointers();
            }

            glBindVertexArray(0);

            glGenBuffers(1, &slot.ubo);
            glBindBuffer(GL_UNIFORM_BUFFER, slot.ubo);
            slot.uboSize = 16 * 1024;
            glBufferData(GL_UNIFORM_BUFFER, slot.uboSize, nullptr, GL_STREAM_DRAW);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    gStreamSlotIdx = 0;
}

static void destroyStreamSlots(const bool bContextLost) noexcept {
    for (StreamSlot& slot : gStreamSlots) {
        if ((!bContextLost) && isGles3()) {
            waitForAndDeleteFence(slot.fence);
        }

        deleteVertexArray(slot.vao, bContextLost);
        deleteBuffer(slot.ubo, bContextLost);
        deleteBuffer(slot.vbo, bContextLost);
        slot = {};
    }

    gStreamSlotIdx = 0;
}

// OpenGL ES 3.0: uploads data to a buffer the GPU no longer uses (its fence was waited on), growing it if needed
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

// The corners of the quads drawn by the post programs, as a 4 vertex triangle strip: top left, top right, bottom left, bottom right
static void initQuadVertices() noexcept {
    static constexpr float CORNERS[8] = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };

    glGenBuffers(1, &gQuadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, gQuadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(CORNERS), CORNERS, GL_STATIC_DRAW);

    if (isGles3()) {
        glGenVertexArrays(1, &gQuadVao);
        glBindVertexArray(gQuadVao);
        glEnableVertexAttribArray(GShaders::ATTRIB_CORNER);
        glVertexAttribPointer(GShaders::ATTRIB_CORNER, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindVertexArray(0);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void bindQuadVertices() noexcept {
    if (isGles3()) {
        glBindVertexArray(gQuadVao);
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, gQuadVbo);
        glVertexAttribPointer(GShaders::ATTRIB_CORNER, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        setNumEnabledVertexAttribs(1);
    }
}

static void unbindQuadVertices() noexcept {
    if (isGles3()) {
        glBindVertexArray(0);
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
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

// OpenGL ES 3.0: packs the frame's uniforms for upload: the first block is zeroed (bound before any uniforms are set), the frame's ones
// follow it.
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

// OpenGL ES 2.0: changes the draw uniforms in effect (the programs get them before their next draw)
static void setDrawUniforms(const VShaderUniforms_Draw& uniforms) noexcept {
    gpDrawUniforms = &uniforms;
    ++gDrawUniformsSerial;
}

// OpenGL ES 2.0: gives a draw program the draw uniforms in effect, if it doesn't have them already
static void applyDrawUniforms(const GProgram program) noexcept {
    const uint32_t programIdx = (uint32_t) program;
    ASSERT(programIdx < NUM_DRAW_PROGRAMS);

    if (gProgramUniformsSerial[programIdx] == gDrawUniformsSerial)
        return;

    gProgramUniformsSerial[programIdx] = gDrawUniformsSerial;
    const GProgramInfo& info = GShaders::get(program);
    const VShaderUniforms_Draw& uniforms = *gpDrawUniforms;

    if (info.uMvpMatrix >= 0) {
        glUniformMatrix4fv(info.uMvpMatrix, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&uniforms.mvpMatrix));
    }

    if (info.uNdcToPsxScale >= 0) {
        glUniform2f(info.uNdcToPsxScale, uniforms.ndcToPsxScaleX, uniforms.ndcToPsxScaleY);
    }

    if (info.uPsxNdcOffset >= 0) {
        glUniform2f(info.uPsxNdcOffset, uniforms.psxNdcOffsetX, uniforms.psxNdcOffsetY);
    }

    if (info.uInvDrawRes >= 0) {
        glUniform2f(info.uInvDrawRes, uniforms.invDrawResX, uniforms.invDrawResY);
    }
}

// Draws the frame recorded by 'VDrawing' into the frame history texture at the given index
static void drawMainFrame(const VDrawing::FrameDrawData& frameData, const uint32_t historyIdx) noexcept {
    StreamSlot& slot = gStreamSlots[gStreamSlotIdx];
    const GLsizeiptr vertsSize = (GLsizeiptr) frameData.numVerts * (GLsizeiptr) sizeof(VVertex_Draw);
    GLsizeiptr uniformBlockStride = 0;

    // Note: the frame is recorded with instances when instancing is used (see 'GRenderer::init')
    const bool bInstanced = frameData.bInstanced;
    ASSERT(bInstanced == gbInstancing);

    if (isGles3()) {
        // Get the buffers for this frame, waiting until the GPU is done with the frame that used them last (normally long done)
        waitForAndDeleteFence(slot.fence);
        uniformBlockStride = ((GLsizeiptr) sizeof(DrawUniformBlock) + gUboOffsetAlignment - 1) / gUboOffsetAlignment * gUboOffsetAlignment;
        const GLsizeiptr uniformsSize = packFrameUniforms(frameData, uniformBlockStride);
        uploadToStreamBuffer(GL_UNIFORM_BUFFER, slot.ubo, slot.uboSize, gUniformUploadBuffer.data(), uniformsSize);

        if (bInstanced) {
            const GLsizeiptr instancesSize = (GLsizeiptr) frameData.numInstances * (GLsizeiptr) sizeof(VInstance_Draw);
            uploadToStreamBuffer(GL_ARRAY_BUFFER, slot.vbo, slot.vboSize, frameData.pInstances, instancesSize);
        } else {
            uploadToStreamBuffer(GL_ARRAY_BUFFER, slot.vbo, slot.vboSize, frameData.pVerts, vertsSize);
        }
    } else if (vertsSize > 0) {
        // OpenGL ES 2.0 has no fences: the buffer's storage is replaced ('orphaned') every frame, so the driver never waits on the GPU
        glBindBuffer(GL_ARRAY_BUFFER, slot.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertsSize, frameData.pVerts, GL_STREAM_DRAW);
        slot.vboSize = vertsSize;
    }

    // Start the pass: clear everything first, so the GPU never loads the old contents into tile memory
    const GLuint drawFbo = (isRenderbufferMsaa(gMsaaMode)) ? gMsaaFbo : gHistoryFbo[historyIdx];
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

    // Vertices and the uniforms in effect before any are set (zeroed)
    if (isGles3()) {
        glBindVertexArray(slot.vao);
        glBindBuffer(GL_ARRAY_BUFFER, slot.vbo);    // Instancing without EXT_base_instance: the attributes are moved per draw
        glBindBufferRange(GL_UNIFORM_BUFFER, GShaders::DRAW_UNIFORMS_BINDING, slot.ubo, 0, (GLsizeiptr) sizeof(DrawUniformBlock));
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, slot.vbo);
        setDrawVertexAttribPointers();
        setNumEnabledVertexAttribs(GShaders::NUM_DRAW_ATTRIBS);
        setDrawUniforms(gZeroDrawUniforms);
    }

    // Replay the drawing commands
    GLenum primitiveType = GL_TRIANGLES;
    GProgram program = GProgram::Colored;

    for (uint32_t cmdIdx = 0; cmdIdx < frameData.numCmds; ++cmdIdx) {
        const VDrawing::DrawCmd& cmd = frameData.pCmds[cmdIdx];

        switch (cmd.type) {
            case VDrawing::DrawCmdType::SetPipeline: {
                const PipelineState state = getPipelineState((VPipelineType_Main) cmd.arg1);
                program = state.program;
                useProgram(GShaders::get(program).program);
                setBlendMode(state.blendMode);
                setCullFace(state.bCullBackFaces);

                if (gbPerSampleShading) {
                    setSampleShading(state.bPerSampleShading);
                }

                primitiveType = (state.bLines) ? GL_LINES : GL_TRIANGLES;
            }   break;

            case VDrawing::DrawCmdType::SetUniforms: {
                if (isGles3()) {
                    glBindBufferRange(
                        GL_UNIFORM_BUFFER,
                        GShaders::DRAW_UNIFORMS_BINDING,
                        slot.ubo,
                        uniformBlockStride * ((GLintptr) cmd.arg1 + 1),
                        (GLsizeiptr) sizeof(DrawUniformBlock)
                    );
                } else {
                    setDrawUniforms(frameData.pUniforms[cmd.arg1]);
                }
            }   break;

            case VDrawing::DrawCmdType::Draw: {
                if (bInstanced) {
                    // One instance per primitive: count and offset are in instances. Lines are 2 vertices, the rest 4 (a strip).
                    const bool bLines = (primitiveType == GL_LINES);
                    const GLenum mode = (bLines) ? GL_LINES : GL_TRIANGLE_STRIP;
                    const GLsizei vertsPerInstance = (bLines) ? 2 : 4;

                    if (gbBaseInstance) {
                        glDrawArraysInstancedBaseInstanceEXT(mode, 0, vertsPerInstance, (GLsizei) cmd.arg1, cmd.arg2);
                    } else {
                        setDrawInstanceAttribPointers(cmd.arg2);
                        glDrawArraysInstanced(mode, 0, vertsPerInstance, (GLsizei) cmd.arg1);
                    }
                } else {
                    if (!isGles3()) {
                        applyDrawUniforms(program);
                    }

                    glDrawArrays(primitiveType, (GLint) cmd.arg2, (GLsizei) cmd.arg1);
                }
            }   break;
        }
    }

    if (isGles3()) {
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        setNumEnabledVertexAttribs(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    setBlendMode(BlendMode::None);
    setCullFace(false);

    if (gbPerSampleShading) {
        setSampleShading(false);
    }

    // MSAA with a renderbuffer: resolve into the frame's texture right away (no scissor, the whole framebuffer: ANGLE does this in the
    // render pass), then drop the samples so they are never written to memory.
    if (isRenderbufferMsaa(gMsaaMode)) {
        setScissorTest(false);
        resolveMultisample(gMsaaMode, gMsaaFbo, gHistoryFbo[historyIdx], (GLint) gHistoryW, (GLint) gHistoryH);
        discardColorAttachment(gMsaaFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // OpenGL ES 3.0: the GPU is done with this slot's buffers once it is past this point
    if (isGles3()) {
        slot.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }

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

    if (isGles3()) {
        glBindSampler(unit, (bLinearFilter) ? gSamplerLinear : gSamplerNearest);
    } else {
        // No sampler objects: the filter is part of the texture
        const GLint filter = (bLinearFilter) ? GL_LINEAR : GL_NEAREST;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    }
}

// Draws a quad with a post program: from 1 or 2 source textures
static void drawPostQuad(
    const GProgram program,
    const float posRect[4],
    const float uvRect[4],
    const GLuint srcTex,
    const GLuint srcTex2,
    const float lerp,
    const bool bLinearFilter
) noexcept {
    const GProgramInfo& programInfo = GShaders::get(program);
    useProgram(programInfo.program);
    setBlendMode(BlendMode::None);
    setCullFace(false);

    if (programInfo.uPosRect >= 0) {
        glUniform4f(programInfo.uPosRect, posRect[0], posRect[1], posRect[2], posRect[3]);
    }

    if (programInfo.uUvRect >= 0) {
        glUniform4f(programInfo.uUvRect, uvRect[0], uvRect[1], uvRect[2], uvRect[3]);
    }

    if (programInfo.uLerp >= 0) {
        glUniform1f(programInfo.uLerp, lerp);
    }

    // Note: integer textures (the PSX framebuffer with OpenGL ES 3.0) must not be filtered
    bindPostTexture(GShaders::TEX_UNIT_SRC, srcTex, bLinearFilter);

    if (srcTex2) {
        bindPostTexture(GShaders::TEX_UNIT_SRC2, srcTex2, bLinearFilter);
    }

    bindQuadVertices();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    unbindQuadVertices();
    unbindPostTextures();
}

// Draws over the whole screen with a post program. The source images have their first row at the top (see the header).
void drawFullscreenPost(const GProgram program, const GLuint srcTex, const GLuint srcTex2, const float lerp) noexcept {
    static constexpr float POS_RECT[4] = { -1.0f, 1.0f, 1.0f, -1.0f };
    static constexpr float UV_RECT[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
    drawPostQuad(program, POS_RECT, UV_RECT, srcTex, srcTex2, lerp, false);
}

void drawQuad(
    const GProgram program,
    const float posRect[4],
    const float uvRect[4],
    const GLuint texture,
    const bool bLinearFilter
) noexcept {
    drawPostQuad(program, posRect, uvRect, texture, 0, 0.0f, bLinearFilter);
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

    // OpenGL ES 2.0: no vertex attributes enabled to begin with (a new or remade context has none)
    if (!isGles3()) {
        for (GLuint attrib = 0; attrib < GShaders::NUM_DRAW_ATTRIBS; ++attrib) {
            glDisableVertexAttribArray(attrib);
        }

        gNumEnabledAttribs = 0;
    }
}

// Tells if program binaries can be used: OpenGL ES 3.0, or OES_get_program_binary. Either way the driver must have a binary format.
static bool canUseProgramBinaries() noexcept {
    const bool bHasApi = (isGles3() || (GLAD_GL_OES_get_program_binary && glProgramBinaryOES && glGetProgramBinaryOES));

    if (!bHasApi)
        return false;

    // Note: the same enum as GL_NUM_PROGRAM_BINARY_FORMATS_OES
    GLint numBinaryFormats = 0;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &numBinaryFormats);
    clearGlErrors();
    return (numBinaryFormats > 0);
}

static void initGlResources() noexcept {
    initGlState();

    // Shaders: binaries from the cache where possible
    const bool bUseProgramBinaries = canUseProgramBinaries();

    const std::string driverId = std::string((const char*) glGetString(GL_VENDOR)) + "|" +
        (const char*) glGetString(GL_RENDERER) + "|" +
        (const char*) glGetString(GL_VERSION) + "|" +
        (const char*) glGetString(GL_SHADING_LANGUAGE_VERSION);

    GShaderCache::init(driverId, bUseProgramBinaries);

    GShaderOptions shaderOptions = {};
    shaderOptions.api = gApi;
    shaderOptions.bUse16BitShading = (!Config::gbUseVulkan32BitShading);
    shaderOptions.bUseProgramBinaries = bUseProgramBinaries;
    shaderOptions.bHighpFragment = gCaps.bHighpFragment;
    shaderOptions.bMakeDrawPrograms = true;
    shaderOptions.bInstancing = gbInstancing;
    shaderOptions.vramHiChannel = (gVramFormat.format == GL_LUMINANCE_ALPHA) ? 'a' : 'g';
    GShaders::init(shaderOptions);
    invalidateStateCache();

    for (uint32_t& serial : gProgramUniformsSerial) {
        serial = 0;
    }

    ++gDrawUniformsSerial;  // New programs: none has the draw uniforms
    gpDrawUniforms = &gZeroDrawUniforms;

    // Samplers for the post programs (OpenGL ES 3.0)
    if (isGles3()) {
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
    }

    // Textures, buffers and framebuffers
    initVram();
    rebuildGammaAdjustTex();

    for (GLuint& texture : gPsxFbTex) {
        texture = createTexture2d(gVramFormat, Video::ORIG_DRAW_RES_X, Video::ORIG_DRAW_RES_Y);
    }

    gPsxFbTexIdx = 0;
    initStreamSlots();
    initQuadVertices();

    // The frame history is made at the start of the next frame (needs the screen size)
    gScreenW = 0;
    gScreenH = 0;
}

static void destroyGlResources(const bool bContextLost) noexcept {
    destroyFrameHistory(bContextLost);
    destroyStreamSlots(bContextLost);
    deleteVertexArray(gQuadVao, bContextLost);
    deleteBuffer(gQuadVbo, bContextLost);

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
    gNumEnabledAttribs = 0;
    gScreenW = 0;
    gScreenH = 0;
    invalidateStateCache();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the renderer for the given API path: the GL context must be current and the GL functions loaded.
// Returns 'false' if the renderer can't run on this device (OpenGL ES 2.0 without 'highp' floats in fragment shaders).
//------------------------------------------------------------------------------------------------------------------------------------------
bool init(SDL_Window* const pWindow, const GApi api) noexcept {
    ASSERT(pWindow);
    ASSERT(!gbInitialized);

    gApi = api;

    if (!determineCaps())
        return false;

    gpWindow = pWindow;
    gbAppInBackground = false;
    gbIsRendering = false;
    gbSkipNextFrame = false;
    gbSkipNextFramePresent = false;

    fitPsxVramToMaxTextureSize();
    VRenderer::updateCoordSysInfo(0, 0);    // Invalid until the first frame
    VDrawing::initForGles(gbInstancing);
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
        gGammaLutTex = createTexture2d(gGammaLutFormat, 256, 1);
    }

    bindTextureForUpload(gGammaLutTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, gGammaLutFormat.format, gGammaLutFormat.type, Video::gGammaTbl.remapTbl8);

    // Stays bound to its texture unit for good
    glActiveTexture(GL_TEXTURE0 + GShaders::TEX_UNIT_GAMMA_LUT);
    glBindTexture(GL_TEXTURE_2D, gGammaLutTex);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Info and helpers for the crossfader, the loading plaque drawer and the display of external surfaces
//------------------------------------------------------------------------------------------------------------------------------------------
GApi getApi() noexcept { return gApi; }
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

//------------------------------------------------------------------------------------------------------------------------------------------
// Whether the OpenGL ES 3.0 path may draw with instancing (the default). Takes effect the next time the renderer is made.
//------------------------------------------------------------------------------------------------------------------------------------------
void setInstancingAllowed(const bool bAllowed) noexcept {
    gbInstancingAllowed = bAllowed;
}

END_NAMESPACE(GRenderer)

//------------------------------------------------------------------------------------------------------------------------------------------
// Android: the launcher can turn instancing off for the OpenGL ES 3.0 path through JNA (on by default)
//------------------------------------------------------------------------------------------------------------------------------------------
#if ANDROID
extern "C" {
__attribute__((used)) __attribute__((visibility("default")))
void setUseGLES3Instancing(const bool useInstancing) {
    GRenderer::setInstancingAllowed(useInstancing);
}
}
#endif

#endif  // #if PSYDOOM_GLES_RENDERER
