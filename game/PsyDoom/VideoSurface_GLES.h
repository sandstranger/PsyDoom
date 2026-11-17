#pragma once

#include "IVideoSurface.h"

#if PSYDOOM_GLES_RENDERER

#include "Gles/GlesApi.h"

#include <vector>

BEGIN_NAMESPACE(Video)

//------------------------------------------------------------------------------------------------------------------------------------------
// GLES backend specific surface: an RGBA8 texture
//------------------------------------------------------------------------------------------------------------------------------------------
class VideoSurface_GLES : public IVideoSurface {
public:
    VideoSurface_GLES(const uint32_t width, const uint32_t height) noexcept;
    virtual ~VideoSurface_GLES() noexcept override;
    virtual uint32_t getWidth() const noexcept override;
    virtual uint32_t getHeight() const noexcept override;
    virtual void setPixels(const uint32_t* const pSrcPixels) noexcept override;

    inline GLuint getTexture() const noexcept { return mTexture; }

private:
    GLuint                  mTexture;           // Contains the surface pixels
    uint32_t                mWidth;             // Pixel width of the surface
    uint32_t                mHeight;            // Pixel height of the surface
    std::vector<uint32_t>   mGammaPixels;       // Gamma corrected pixels, if gamma correction is on
};

END_NAMESPACE(Video)

#endif  // #if PSYDOOM_GLES_RENDERER
