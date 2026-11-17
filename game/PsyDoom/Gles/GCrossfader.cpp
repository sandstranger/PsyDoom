//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: crossfades between the last two frames drawn by the main render path.
// Works the same way as 'VCrossfader' of the Vulkan renderer: the frame to fade into is drawn but not shown, then the crossfade render
// path mixes it with the frame before it, straight on the screen.
//------------------------------------------------------------------------------------------------------------------------------------------
#include "GCrossfader.h"

#if PSYDOOM_GLES_RENDERER

#include "Asserts.h"
#include "Doom/Base/i_main.h"
#include "GRenderer.h"
#include "PsyDoom/Input.h"
#include "PsyDoom/Utils.h"

BEGIN_NAMESPACE(GCrossfader)

// The frames faded from and to
static GLuint gCrossfadeTex1;
static GLuint gCrossfadeTex2;

//------------------------------------------------------------------------------------------------------------------------------------------
// Called before drawing the frame to be faded into: that frame (the current one, of the main render path) is drawn but not shown, and
// the crossfade render path is used from the next frame on.
//------------------------------------------------------------------------------------------------------------------------------------------
void doPreCrossfadeSetup() noexcept {
    // The frame drawn before (shown on the screen now) and the one being drawn
    const uint32_t historyIdx = GRenderer::getFrameHistoryIdx();
    gCrossfadeTex1 = GRenderer::getFrameHistoryTex(historyIdx + 1);
    gCrossfadeTex2 = GRenderer::getFrameHistoryTex(historyIdx);

    GRenderer::setNextRenderPath(GRenderer::RenderPath::Crossfade);
    GRenderer::gbSkipNextFramePresent = true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Does the crossfade, lasting the given number of vblanks
//------------------------------------------------------------------------------------------------------------------------------------------
void doCrossfade(const int32_t vblanksDuration) noexcept {
    ASSERT(GRenderer::getActiveRenderPath() == GRenderer::RenderPath::Crossfade);

    const int32_t fadeBeginTimeVbl = I_GetTotalVBlanks();
    float percentComplete = 0.0f;
    bool bShouldContinueRendering = true;

    while (bShouldContinueRendering) {
        // This is a new frame drawn
        I_IncDrawnFrameCount();

        // Draw the crossfade, if the frame can be drawn
        if (GRenderer::isRendering() && gCrossfadeTex1 && gCrossfadeTex2) {
            const GProgram program = (GRenderer::gbUsingGammaAdjustThisFrame) ? GProgram::Crossfade_Gamma : GProgram::Crossfade;
            GRenderer::drawFullscreenPost(program, gCrossfadeTex1, gCrossfadeTex2, percentComplete);
        }

        // Done yet?
        const int32_t nowTimeVbl = I_GetTotalVBlanks();
        const int32_t elapsedVbl = nowTimeVbl - fadeBeginTimeVbl;

        if ((elapsedVbl >= vblanksDuration) || Input::isQuitRequested()) {
            bShouldContinueRendering = false;
        } else {
            percentComplete = (float) elapsedVbl / (float) vblanksDuration;

            // Show the frame and start the next
            if (GRenderer::isRendering()) {
                GRenderer::endFrame();
            }

            Utils::doPlatformUpdates();
            Utils::threadYield();
            GRenderer::beginFrame();
        }
    }

    // Show the last frame, then go back to the main render path
    if (GRenderer::isRendering()) {
        GRenderer::endFrame();
    }

    GRenderer::setNextRenderPath(GRenderer::RenderPath::Main);
    GRenderer::beginFrame();

    gCrossfadeTex1 = 0;
    gCrossfadeTex2 = 0;
}

END_NAMESPACE(GCrossfader)

#endif  // #if PSYDOOM_GLES_RENDERER
