#include "VideoSurface_Vulkan.h"

#if PSYDOOM_VULKAN_RENDERER

#include "Asserts.h"
#include "GammaTable.h"
#include "PlayerPrefs.h"
#include "Video.h"

#include <cstring>

BEGIN_NAMESPACE(Video)

//------------------------------------------------------------------------------------------------------------------------------------------
// Attempts to create the surface with the specified dimensions.
// If creation fails then the failure is silent and usage of the surface results in NO-OPs.
//------------------------------------------------------------------------------------------------------------------------------------------
VideoSurface_Vulkan::VideoSurface_Vulkan(vgl::LogicalDevice& device, const uint32_t width, const uint32_t height) noexcept
    : mTexture()
    , mWidth(width)
    , mHeight(height)
    , mbIsReadyForBlit(false)
{
    ASSERT((width > 0) && (height > 0));

    if (!mTexture.initAs2dTexture(device, VK_FORMAT_B8G8R8A8_UNORM, width, height, true)) {  // N.B: 'true' for copyable!
        ASSERT_FAIL("VideoSurface_Vulkan::VideoSurface_Vulkan: initializing the texture failed!");
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Cleans up surface resources
//------------------------------------------------------------------------------------------------------------------------------------------
VideoSurface_Vulkan::~VideoSurface_Vulkan() noexcept = default;

//------------------------------------------------------------------------------------------------------------------------------------------
// Get the dimensions of the surface
//------------------------------------------------------------------------------------------------------------------------------------------
uint32_t VideoSurface_Vulkan::getWidth() const noexcept { return mWidth; }
uint32_t VideoSurface_Vulkan::getHeight() const noexcept { return mHeight; }

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the pixels for the surface.
// Ignores the call if the surface was not successfully initialized.
//------------------------------------------------------------------------------------------------------------------------------------------
void VideoSurface_Vulkan::setPixels(const uint32_t* const pSrcPixels) noexcept {
    ASSERT(pSrcPixels);

    // Abort if the texture wasn't successfully initialized
    if (!mTexture.isValid())
        return;

    // Gamma correct related vars
    const bool bDoGammaCorrection = (PlayerPrefs::gGamma1000 != PlayerPrefs::GAMMA_DEFAULT);
    const uint8_t* const pGammaRemapTbl8 = Video::gGammaTbl.remapTbl8;

    // Do the copy (assuming the texture lock succeeds)
    const uint32_t numPixels = mWidth * mHeight;
    uint32_t* const pDstPixels = reinterpret_cast<uint32_t*>(mTexture.lock());

    if (!pDstPixels)
        return;

    if (!bDoGammaCorrection) {
        // Regular (fast) copy path
        std::memcpy(pDstPixels, pSrcPixels, numPixels * sizeof(uint32_t));
    }
    else {
        // Gamma corrected copy
        for (uint32_t i = 0; i < numPixels; ++i) {
            const uint32_t srcPixel = pSrcPixels[i];
            const uint8_t r = pGammaRemapTbl8[(uint8_t)(srcPixel)];
            const uint8_t g = pGammaRemapTbl8[(uint8_t)(srcPixel >> 8)];
            const uint8_t b = pGammaRemapTbl8[(uint8_t)(srcPixel >> 16)];

            pDstPixels[i] = (srcPixel & 0xFF000000) | (b << 16) | (g << 8) | r;
        }
    }

    // Finish up
    mTexture.unlock();
    mbIsReadyForBlit = false;   // Texture will be shader read only optimal after the transfer!
}

END_NAMESPACE(Video)

#endif  // #if PSYDOOM_VULKAN_RENDERER
