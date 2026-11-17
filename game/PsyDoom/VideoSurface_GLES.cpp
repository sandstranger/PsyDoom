#include "VideoSurface_GLES.h"

#if PSYDOOM_GLES_RENDERER

#include "Asserts.h"
#include "Gles/GRenderer.h"
#include "GammaTable.h"
#include "PlayerPrefs.h"
#include "Video.h"

BEGIN_NAMESPACE(Video)

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the surface with the specified dimensions (the GL context must be current)
//------------------------------------------------------------------------------------------------------------------------------------------
VideoSurface_GLES::VideoSurface_GLES(const uint32_t width, const uint32_t height) noexcept
    : mTexture(0)
    , mWidth(width)
    , mHeight(height)
    , mGammaPixels()
{
    ASSERT((width > 0) && (height > 0));

    glGenTextures(1, &mTexture);
    GRenderer::bindTextureForUpload(mTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, (GLsizei) width, (GLsizei) height);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Cleans up surface resources
//------------------------------------------------------------------------------------------------------------------------------------------
VideoSurface_GLES::~VideoSurface_GLES() noexcept {
    if (mTexture) {
        glDeleteTextures(1, &mTexture);
        mTexture = 0;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Get the dimensions of the surface
//------------------------------------------------------------------------------------------------------------------------------------------
uint32_t VideoSurface_GLES::getWidth() const noexcept { return mWidth; }
uint32_t VideoSurface_GLES::getHeight() const noexcept { return mHeight; }

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the pixels for the surface: XBGR8888, which is RGBA in memory (red in the lowest byte)
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoSurface_GLES::setPixels(const uint32_t* const pSrcPixels) noexcept {
    ASSERT(pSrcPixels);

    if (!mTexture)
        return;

    const uint32_t numPixels = mWidth * mHeight;
    const uint32_t* pUploadPixels = pSrcPixels;

    // Gamma correction is done here on the CPU, as with the Vulkan backend
    if (PlayerPrefs::gGamma1000 != PlayerPrefs::GAMMA_DEFAULT) {
        const uint8_t* const pGammaRemapTbl8 = Video::gGammaTbl.remapTbl8;
        mGammaPixels.resize(numPixels);

        for (uint32_t i = 0; i < numPixels; ++i) {
            const uint32_t srcPixel = pSrcPixels[i];
            const uint8_t r = pGammaRemapTbl8[(uint8_t)(srcPixel)];
            const uint8_t g = pGammaRemapTbl8[(uint8_t)(srcPixel >> 8)];
            const uint8_t b = pGammaRemapTbl8[(uint8_t)(srcPixel >> 16)];

            mGammaPixels[i] = (srcPixel & 0xFF000000) | ((uint32_t) b << 16) | ((uint32_t) g << 8) | r;
        }

        pUploadPixels = mGammaPixels.data();
    }

    GRenderer::bindTextureForUpload(mTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei) mWidth, (GLsizei) mHeight, GL_RGBA, GL_UNSIGNED_BYTE, pUploadPixels);
}

END_NAMESPACE(Video)

#endif  // #if PSYDOOM_GLES_RENDERER
