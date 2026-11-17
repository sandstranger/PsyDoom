#if PSYDOOM_GLES_RENDERER

#include "VideoBackend_GLES.h"

#include "Asserts.h"
#include "Config/Config.h"
#include "FatalErrors.h"
#include "VideoSurface_GLES.h"

#include <algorithm>
#include <SDL.h>

BEGIN_NAMESPACE(Video)

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the backend uninitialized
//------------------------------------------------------------------------------------------------------------------------------------------
VideoBackend_GLES::VideoBackend_GLES() noexcept
    : mpSdlWindow(nullptr)
    , mGlContext(nullptr)
    , mDispExtSurfOldRenderPath(GRenderer::RenderPath::Main)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Ensures everything is cleaned up
//------------------------------------------------------------------------------------------------------------------------------------------
VideoBackend_GLES::~VideoBackend_GLES() noexcept {
    destroyRenderers();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets up the OpenGL attributes for the window: an OpenGL ES 3.0 context, 32-bit color and no depth, stencil or multisampling in the
// window surface. The renderer draws into its own framebuffers (MSAA is done there), the window surface is only ever cleared and
// filled with the finished frame: no memory is wasted on attachments that are never used.
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::prepareForWindowCreate() noexcept {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Get the window create flags required for an SDL video backend window
//------------------------------------------------------------------------------------------------------------------------------------------
uint32_t VideoBackend_GLES::getSdlWindowCreateFlags() noexcept {
    return SDL_WINDOW_OPENGL;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the OpenGL ES 3.0 context and the renderer.
// Returns 'false' if OpenGL ES 3.0 is not available, in which case another backend must be used (the window is still usable by SDL).
//------------------------------------------------------------------------------------------------------------------------------------------
bool VideoBackend_GLES::tryInitRenderers(SDL_Window* const pSdlWindow) noexcept {
    ASSERT(pSdlWindow);
    ASSERT(!mGlContext);
    mpSdlWindow = pSdlWindow;

    mGlContext = SDL_GL_CreateContext(pSdlWindow);

    if (!mGlContext) {
        SDL_Log("PsyDoom GLES: can't create an OpenGL ES 3.0 context: %s", SDL_GetError());
        mpSdlWindow = nullptr;
        return false;
    }

    const bool bContextOk = (
        (SDL_GL_MakeCurrent(pSdlWindow, mGlContext) == 0) &&
        gladLoadGLES2Loader((GLADloadproc) SDL_GL_GetProcAddress) &&
        (GLVersion.major >= 3)
    );

    if (!bContextOk) {
        SDL_Log("PsyDoom GLES: OpenGL ES 3.0 is not available");
        SDL_GL_MakeCurrent(pSdlWindow, nullptr);
        SDL_GL_DeleteContext(mGlContext);
        mGlContext = nullptr;
        mpSdlWindow = nullptr;
        return false;
    }

    SDL_GL_SetSwapInterval((Config::gbEnableVSync) ? 1 : 0);

    if (!GRenderer::init(pSdlWindow)) {
        SDL_GL_MakeCurrent(pSdlWindow, nullptr);
        SDL_GL_DeleteContext(mGlContext);
        mGlContext = nullptr;
        mpSdlWindow = nullptr;
        return false;
    }

    GRenderer::beginFrame();
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the renderers for this backend, which must succeed (see 'tryInitRenderers' for an attempt that can fail)
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::initRenderers(SDL_Window* const pSdlWindow) noexcept {
    if (!tryInitRenderers(pSdlWindow)) {
        FatalErrors::raise("Failed to initialize the OpenGL ES 3.0 renderer!");
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Destroys the renderer and the GL context
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::destroyRenderers() noexcept {
    if (!mGlContext)
        return;

    GRenderer::destroy();
    SDL_GL_MakeCurrent(mpSdlWindow, nullptr);
    SDL_GL_DeleteContext(mGlContext);
    mGlContext = nullptr;
    mpSdlWindow = nullptr;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Called when SDL had to make a new GL context because the old one was lost (Android): all GL objects must be made again
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::onContextLost() noexcept {
    if (!mGlContext)
        return;

    mGlContext = SDL_GL_GetCurrentContext();
    GRenderer::onContextLost();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Displays the output from the renderer (classic, or new) on the screen and starts the next frame
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::displayFramebuffer() noexcept {
    GRenderer::endFrame();
    GRenderer::beginFrame();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Performs setup prior to displaying external surfaces: frames are started manually by 'displayExternalSurface' from here on
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::beginExternalSurfaceDisplay() noexcept {
    mDispExtSurfOldRenderPath = GRenderer::getActiveRenderPath();

    if (GRenderer::isRendering()) {
        GRenderer::endFrame();
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Performs cleanup once done displaying external surfaces: back to automatically started frames and the previous render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::endExternalSurfaceDisplay() noexcept {
    GRenderer::setNextRenderPath(mDispExtSurfOldRenderPath);
    GRenderer::beginFrame();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Displays the specified surface on the screen at the given area (in pixels from the top left), the rest of the screen black
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::displayExternalSurface(
    IVideoSurface& surface,
    const int32_t displayX,
    const int32_t displayY,
    const uint32_t displayW,
    const uint32_t displayH,
    const bool bUseFiltering
) noexcept {
    // Must be a GLES video surface!
    ASSERT(dynamic_cast<VideoSurface_GLES*>(&surface));
    VideoSurface_GLES& glesSurface = static_cast<VideoSurface_GLES&>(surface);

    // Note: it's assumed that no frame is currently active, see 'beginExternalSurfaceDisplay'
    GRenderer::setNextRenderPath(GRenderer::RenderPath::Blit);

    if (!GRenderer::beginFrame())
        return;

    const float screenW = (float) GRenderer::getScreenWidth();
    const float screenH = (float) GRenderer::getScreenHeight();

    if ((displayW > 0) && (displayH > 0) && (screenW > 0) && (screenH > 0) && glesSurface.getTexture()) {
        // Note: OpenGL NDC has y up, the area is given from the top. Anything off the screen is clipped by the GPU.
        const float posRect[4] = {
            ((float) displayX / screenW) * 2.0f - 1.0f,
            1.0f - ((float) displayY / screenH) * 2.0f,
            ((float)(displayX + (int32_t) displayW) / screenW) * 2.0f - 1.0f,
            1.0f - ((float)(displayY + (int32_t) displayH) / screenH) * 2.0f
        };

        const float uvRect[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
        GRenderer::drawQuad(GProgram::Quad, posRect, uvRect, glesSurface.getTexture(), bUseFiltering);
    }

    GRenderer::endFrame();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Gives the size of the window surface in pixels
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoBackend_GLES::getScreenSizeInPixels(uint32_t& width, uint32_t& height) noexcept {
    int drawableW = 0;
    int drawableH = 0;

    if (mpSdlWindow) {
        SDL_GL_GetDrawableSize(mpSdlWindow, &drawableW, &drawableH);
    }

    width = (uint32_t) std::max(drawableW, 0);
    height = (uint32_t) std::max(drawableH, 0);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates a GLES video surface; fails if the renderer is not initialized
//------------------------------------------------------------------------------------------------------------------------------------------
std::unique_ptr<IVideoSurface> VideoBackend_GLES::createSurface(const uint32_t width, const uint32_t height) noexcept {
    return (mGlContext) ?
        std::make_unique<VideoSurface_GLES>(width, height) :
        std::unique_ptr<IVideoSurface>();
}

END_NAMESPACE(Video)

#endif  // #if PSYDOOM_GLES_RENDERER
