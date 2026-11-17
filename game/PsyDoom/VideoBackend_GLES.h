#pragma once

#if PSYDOOM_GLES_RENDERER

#include "Gles/GRenderer.h"
#include "IVideoBackend.h"

typedef void* SDL_GLContext;

BEGIN_NAMESPACE(Video)

//------------------------------------------------------------------------------------------------------------------------------------------
// A video backend that uses OpenGL ES 3.0 to display the current framebuffer to the screen.
// Supports both the classic PlayStation renderer and the new hardware accelerated renderer (the GLES version of the Vulkan renderer).
//------------------------------------------------------------------------------------------------------------------------------------------
class VideoBackend_GLES : public IVideoBackend {
public:
    VideoBackend_GLES() noexcept;
    virtual ~VideoBackend_GLES() noexcept;

    virtual void prepareForWindowCreate() noexcept override;
    virtual uint32_t getSdlWindowCreateFlags() noexcept override;
    virtual void initRenderers(SDL_Window* const pSdlWindow) noexcept override;
    virtual void destroyRenderers() noexcept override;
    virtual void displayFramebuffer() noexcept override;

    virtual void beginExternalSurfaceDisplay() noexcept override;
    virtual void endExternalSurfaceDisplay() noexcept override;

    virtual void displayExternalSurface(
        IVideoSurface& surface,
        const int32_t displayX,
        const int32_t displayY,
        const uint32_t displayW,
        const uint32_t displayH,
        const bool bUseFiltering
    ) noexcept override;

    virtual void getScreenSizeInPixels(uint32_t& width, uint32_t& height) noexcept override;
    [[nodiscard]] virtual std::unique_ptr<IVideoSurface> createSurface(const uint32_t width, const uint32_t height) noexcept override;

    bool tryInitRenderers(SDL_Window* const pSdlWindow) noexcept;
    void onContextLost() noexcept;

private:
    SDL_Window*             mpSdlWindow;                    // The SDL window used
    SDL_GLContext           mGlContext;                     // The OpenGL ES 3.0 context
    GRenderer::RenderPath   mDispExtSurfOldRenderPath;      // The render path used prior to displaying external surfaces (restored after)
};

END_NAMESPACE(Video)

#endif  // PSYDOOM_GLES_RENDERER
