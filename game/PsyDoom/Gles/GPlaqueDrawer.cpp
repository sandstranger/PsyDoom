//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: draws loading plaques over the last frame of the main render path.
// Works the same way as 'VPlaqueDrawer' of the Vulkan renderer.
//------------------------------------------------------------------------------------------------------------------------------------------
#include "GPlaqueDrawer.h"

#if PSYDOOM_GLES_RENDERER

#include "Asserts.h"
#include "Doom/Base/i_main.h"
#include "Doom/Base/i_texcache.h"
#include "Doom/Renderer/r_data.h"
#include "Doom/RendererVk/rv_utils.h"
#include "Gpu.h"
#include "GRenderer.h"
#include "PsyDoom/PsxVm.h"
#include "PsyDoom/Vulkan/VDrawing.h"

#include <vector>

BEGIN_NAMESPACE(GPlaqueDrawer)

//------------------------------------------------------------------------------------------------------------------------------------------
// Makes an RGBA8 texture of the plaque from PSX VRAM (8bpp + CLUT), with fully transparent texels where the PSX would mask
//------------------------------------------------------------------------------------------------------------------------------------------
static GLuint createPlaqueTex(const texture_t& tex, const int16_t clutId) noexcept {
    ASSERT(tex.ppTexCacheEntries);

    uint16_t texX, texY, texW, texH;
    RV_GetTexWinXyWh(tex, texX, texY, texW, texH);

    uint16_t clutX, clutY;
    RV_ClutIdToClutXy(clutId, clutX, clutY);

    const uint16_t* const pVram = PsxVm::gGpu.pRam;
    const uint32_t vramWidth = PsxVm::gGpu.ramPixelW;
    const uint16_t* const pClut = pVram + ((clutY * vramWidth) + clutX);

    // Note: RGBA byte order (red in the lowest byte)
    std::vector<uint32_t> pixels((size_t) texW * texH);

    for (uint32_t y = 0; y < texH; ++y) {
        for (uint32_t x = 0; x < texW; ++x) {
            const uint16_t srcVram = pVram[(texY + y) * vramWidth + (texX + x) / 2];   // Note: divide x by '2' to convert to 16bpp pixel coords
            const uint16_t srcColorIdx = (x & 1) ? srcVram >> 8 : srcVram & 0xFFu;
            const uint16_t srcPixel = pClut[srcColorIdx];
            uint32_t& dstPixel = pixels[(size_t) texW * y + x];

            if (srcPixel != 0) {
                const uint32_t srcR = ((srcPixel >>  0) & 0x1F) << 3;
                const uint32_t srcG = ((srcPixel >>  5) & 0x1F) << 3;
                const uint32_t srcB = ((srcPixel >> 10) & 0x1F) << 3;
                dstPixel = (srcR << 0) | (srcG << 8) | (srcB << 16) | 0xFF000000u;
            } else {
                dstPixel = 0;
            }
        }
    }

    const GLuint texture = GRenderer::createRgba8Texture(texW, texH);
    GRenderer::bindTextureForUpload(texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texW, texH, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    return texture;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Draws the loading plaque over the last frame drawn and shows it
//------------------------------------------------------------------------------------------------------------------------------------------
void drawPlaque(texture_t& plaqueTex, const int16_t plaqueX, const int16_t plaqueY, const int16_t clutId) noexcept {
    // This is a new frame drawn
    I_IncDrawnFrameCount();

    // Both frames of the main render path must have been drawn: the last one is the background.
    // Note: stop if frames can't be drawn at all right now (app in the background).
    while (!GRenderer::didRenderToAllFrameHistory()) {
        if (GRenderer::isRendering()) {
            GRenderer::endFrame();
        }

        GRenderer::setNextRenderPath(GRenderer::RenderPath::Main);

        if (!GRenderer::beginFrame())
            break;
    }

    // Make the plaque texture and get the background: the frame drawn before the current one
    I_CacheTex(plaqueTex);
    const GLuint plaqueGlTex = createPlaqueTex(plaqueTex, clutId);
    const GLuint backgroundTex = GRenderer::getFrameHistoryTex(GRenderer::getFrameHistoryIdx() + 1);

    // Finish the current frame. If nothing was drawn in it (main render path), don't show it: it would flash black.
    bool bShouldPresentPlaque = true;

    if (GRenderer::isRendering()) {
        if ((GRenderer::getActiveRenderPath() == GRenderer::RenderPath::Main) && (!VDrawing::hasIssuedDrawCmds())) {
            GRenderer::gbSkipNextFramePresent = true;
            bShouldPresentPlaque = false;
        }

        GRenderer::endFrame();
    }

    // Draw and show the plaque over the background
    if (bShouldPresentPlaque) {
        GRenderer::setNextRenderPath(GRenderer::RenderPath::LoadingPlaque);
        GRenderer::beginFrame();

        if (GRenderer::isRendering()) {
            const bool bGammaAdjust = GRenderer::gbUsingGammaAdjustThisFrame;

            // The background, filling the screen. Its first row is the top (see 'GRenderer').
            const float bgPosRect[4] = { -1.0f, 1.0f, 1.0f, -1.0f };
            const float bgUvRect[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
            GRenderer::drawQuad((bGammaAdjust) ? GProgram::Quad_Gamma : GProgram::Quad, bgPosRect, bgUvRect, backgroundTex, false);

            // The plaque, in UI coordinates: transformed to NDC as the Vulkan renderer does, then y flipped (OpenGL NDC has y up)
            const Matrix4f uiTransforms = VDrawing::computeTransformMatrixForUI(true);

            float plaqueTL[3] = { (float) plaqueX, (float) plaqueY, 0.0f };
            float plaqueBR[3] = { (float) plaqueX + (float) plaqueTex.width, (float) plaqueY + (float) plaqueTex.height, 0.0f };
            uiTransforms.transform3d(plaqueTL, plaqueTL);
            uiTransforms.transform3d(plaqueBR, plaqueBR);

            const float plaquePosRect[4] = { plaqueTL[0], -plaqueTL[1], plaqueBR[0], -plaqueBR[1] };
            const float plaqueUvRect[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
            GRenderer::drawQuad((bGammaAdjust) ? GProgram::QuadMasked_Gamma : GProgram::QuadMasked, plaquePosRect, plaqueUvRect, plaqueGlTex, false);

            GRenderer::endFrame();
        }
    }

    // Cleanup and go back to the main render path
    glDeleteTextures(1, &plaqueGlTex);
    GRenderer::setNextRenderPath(GRenderer::RenderPath::Main);
    GRenderer::beginFrame();
}

END_NAMESPACE(GPlaqueDrawer)

#endif  // #if PSYDOOM_GLES_RENDERER
