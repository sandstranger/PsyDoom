#pragma once

#if PSYDOOM_GLES_RENDERER

#include "GlesApi.h"
#include "GShaders.h"
#include "Macros.h"

#include <cstdint>

struct SDL_Window;

//------------------------------------------------------------------------------------------------------------------------------------------
// The OpenGL ES 3.0 renderer, core module: a second backend for the new renderer, with the same features as the Vulkan one.
// The renderer frontend is shared: the game draws through 'VDrawing' (the 'RendererVk' code, UI drawing etc.) and this module replays
// what was drawn at the end of the frame, like the Vulkan renderer records its command buffer then.
//
// It is written for tile based GPUs (Mali, Adreno, PowerVR):
//  - Every render pass starts with a clear of all of its attachments, so tile memory is never loaded from memory, and there is no depth
//    or stencil buffer at all (also none in the window surface).
//  - MSAA through EXT_multisampled_render_to_texture (or the IMG version): the samples only exist in tile memory and are resolved as the
//    tiles are written out, so MSAA costs no memory bandwidth. Without it: a multisampled renderbuffer resolved with a blit, then
//    invalidated so its samples are never written to memory.
//  - Each frame is a fixed sequence of passes, each to one framebuffer and done once: (1) PSX VRAM updates, (2) the frame, (3) the frame
//    shown on the screen with gamma adjustment. No framebuffer switches in between, no reads back to the CPU, no glFinish.
//  - Vertices and uniforms are uploaded once per frame into a ring of buffers protected by fences: no buffer the GPU is still reading
//    is ever written (no driver copies or stalls).
//  - PSX VRAM (up to 128 MiB) is never written by the CPU while frames using it are in flight, which would make Mali drivers copy the
//    whole texture ('resource ghosting'): the updated areas are uploaded to a small staging texture and copied into VRAM by the GPU,
//    in order with the frames reading it.
//  - Program binaries are cached (GShaderCache), so shaders are only compiled once per driver.
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

bool init(SDL_Window* const pWindow) noexcept;
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

uint32_t getScreenWidth() noexcept;
uint32_t getScreenHeight() noexcept;
uint32_t getFrameHistoryIdx() noexcept;
GLuint getFrameHistoryTex(const uint32_t idx) noexcept;
bool didRenderToAllFrameHistory() noexcept;
void bindTextureForUpload(const GLuint texture) noexcept;

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
