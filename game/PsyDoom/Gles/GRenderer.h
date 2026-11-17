#pragma once

#if PSYDOOM_GLES_RENDERER

#include "GlesApi.h"
#include "GShaders.h"
#include "Macros.h"

#include <cstdint>

struct SDL_Window;

//------------------------------------------------------------------------------------------------------------------------------------------
// The OpenGL ES renderer, core module: a second backend for the new renderer, with the same features as the Vulkan one.
// The renderer frontend is shared: the game draws through 'VDrawing' (the 'RendererVk' code, UI drawing etc.) and this module replays
// what was drawn at the end of the frame, like the Vulkan renderer records its command buffer then.
//
// Two API paths, chosen when the renderer is made ('GApi'):
//  - OpenGL ES 3.0: integer textures (PSX VRAM is R16UI), uniform blocks, VAOs, samplers, fences, blits.
//  - OpenGL ES 2.0: core 2.0 plus what extensions there are. PSX VRAM is a 2 bytes per texel texture (RG8 with EXT_texture_rg, else
//    luminance alpha) decoded with float arithmetic in the shaders, which needs 'highp' floats in fragment shaders (the renderer can't
//    run without them). Plain uniforms, vertex buffers orphaned every frame instead of fences, VRAM updates copied in with
//    glCopyTexSubImage2D (RG8 only, otherwise uploaded directly).
//
// It is written for tile based GPUs (Mali, Adreno, PowerVR), with both paths:
//  - Every render pass starts with a clear of all of its attachments, so tile memory is never loaded from memory, and there is no depth
//    or stencil buffer at all (also none in the window surface).
//  - MSAA, checked through the extensions: EXT_multisampled_render_to_texture (or the IMG version) first, where the samples only exist
//    in tile memory and are resolved as the tiles are written out, so MSAA costs no memory bandwidth. Otherwise a multisampled
//    renderbuffer (OpenGL ES 3.0, or ANGLE_framebuffer_multisample, NV_framebuffer_multisample, APPLE_framebuffer_multisample) resolved
//    right after drawing, then invalidated/discarded so its samples are never written to memory. ANGLE (Vulkan) does the resolve in the
//    render pass itself then, so it is just as cheap; ANGLE also has EXT_multisampled_render_to_texture, which comes first.
//  - Each frame is a fixed sequence of passes, each to one framebuffer and done once: (1) PSX VRAM updates, (2) the frame, (3) the frame
//    shown on the screen with gamma adjustment. No framebuffer switches in between, no reads back to the CPU, no glFinish.
//  - OpenGL ES 3.0 draws the main render path with instancing: one instance per primitive ('VInstance_Draw', recorded by 'VDrawing'),
//    a quad as a 4 vertex strip with its corners picked by 'gl_VertexID'. A quad is then 112 bytes instead of 6 vertices of 40 bytes,
//    with 4 vertex shader runs instead of 6, and still the same triangles in the same order (no depth buffer: the order matters).
//    The draws start at their first instance with EXT_base_instance, otherwise the instance attributes are moved to it.
//  - Vertices and uniforms are uploaded once per frame: OpenGL ES 3.0 into a ring of buffers protected by fences, OpenGL ES 2.0 into
//    buffers whose storage is replaced every frame. No buffer the GPU is still reading is ever written (no driver copies or stalls).
//  - PSX VRAM (up to 128 MiB) is never written by the CPU while frames using it are in flight, which would make Mali drivers copy the
//    whole texture ('resource ghosting'): the updated areas are uploaded to a small staging texture and copied into VRAM by the GPU,
//    in order with the frames reading it.
//  - Program binaries are cached (GShaderCache, OES_get_program_binary with OpenGL ES 2.0), so shaders are only compiled once per driver.
//
// The images drawn by the main render path keep the Vulkan layout: their first row is the top of the image (the same NDC and viewport
// as with Vulkan, the triangle winding is the other way around in OpenGL), they are flipped when shown on the screen.
//------------------------------------------------------------------------------------------------------------------------------------------
BEGIN_NAMESPACE(GRenderer)

// The render paths: the same as those of the Vulkan renderer
enum class RenderPath : uint8_t {
    Psx,                // Classic renderer: shows the PSX framebuffer
    Main,               // The new renderer
    Crossfade,          // Crossfades between two frames of the main render path
    LoadingPlaque,      // Shows a loading plaque over the last frame of the main render path
    Blit                // Shows external surfaces (movies, logos)
};

extern bool gbSkipNextFramePresent;
extern bool gbSkipNextFrame;
extern bool gbUsingGammaAdjustThisFrame;

bool init(SDL_Window* const pWindow, const GApi api) noexcept;
void destroy() noexcept;
void onContextLost() noexcept;
void onAppEnterBackground() noexcept;
void onAppEnterForeground() noexcept;

bool beginFrame() noexcept;
bool isRendering() noexcept;
void endFrame() noexcept;

RenderPath getActiveRenderPath() noexcept;
RenderPath getNextRenderPath() noexcept;
void setNextRenderPath(const RenderPath renderPath) noexcept;
bool isUsingPsxRenderPath() noexcept;
bool isNextRenderPathPsx() noexcept;
void switchToPsxRenderPath() noexcept;
void switchToMainRenderPath() noexcept;

void pushPsxVramUpdates(const uint16_t rectLx, const uint16_t rectRx, const uint16_t rectTy, const uint16_t rectBy) noexcept;
void rebuildGammaAdjustTex() noexcept;

GApi getApi() noexcept;
uint32_t getScreenWidth() noexcept;
uint32_t getScreenHeight() noexcept;
uint32_t getFrameHistoryIdx() noexcept;
GLuint getFrameHistoryTex(const uint32_t idx) noexcept;
bool didRenderToAllFrameHistory() noexcept;
void bindTextureForUpload(const GLuint texture) noexcept;
GLuint createRgba8Texture(const uint32_t width, const uint32_t height) noexcept;
void setInstancingAllowed(const bool bAllowed) noexcept;

void drawFullscreenPost(const GProgram program, const GLuint srcTex, const GLuint srcTex2, const float lerp) noexcept;

void drawQuad(
    const GProgram program,
    const float posRect[4],
    const float uvRect[4],
    const GLuint texture,
    const bool bLinearFilter
) noexcept;

END_NAMESPACE(GRenderer)

#endif  // #if PSYDOOM_GLES_RENDERER
