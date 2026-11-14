//------------------------------------------------------------------------------------------------------------------------------------------
// Various high level logic and utility functions for the new Vulkan world renderer.
// Note that this renderer re-uses some of the global state and functionality from the original 'r_' renderer modules.
//------------------------------------------------------------------------------------------------------------------------------------------
#if PSYDOOM_VULKAN_RENDERER

#include "rv_main.h"

#include "Doom/Base/i_main.h"
#include "Doom/Base/w_wad.h"
#include "Doom/Game/doomdata.h"
#include "Doom/Game/g_game.h"
#include "Doom/Game/p_user.h"
#include "Doom/Renderer/r_data.h"
#include "Doom/Renderer/r_local.h"
#include "Doom/Renderer/r_main.h"
#include "Doom/Renderer/r_sky.h"
#include "Doom/Renderer/r_things.h"
#include "Doom/Base/i_texcache.h"
#include "Gpu.h"
#include "PsyDoom/Config/Config.h"
#include "PsyDoom/PlayerPrefs.h"
#include "PsyDoom/PsxVm.h"
#include "PsyDoom/Utils.h"
#include "PsyDoom/Vulkan/VDrawing.h"
#include "PsyDoom/Vulkan/VRenderer.h"
#include "PsyDoom/Vulkan/VTypes.h"
#include "PsyQ/LIBGPU.h"
#include "rv_bsp.h"
#include "rv_flats.h"
#include "rv_occlusion.h"
#include "rv_sky.h"
#include "rv_sprites.h"
#include "rv_utils.h"
#include "rv_walls.h"

float               gViewXf, gViewYf, gViewZf;  // View position in floating point format
float               gViewAnglef;                // View angle in radians (float)
float               gViewCosf, gViewSinf;       // Sin and cosine for view angle
uint16_t            gClutX, gClutY;             // X and Y position in VRAM for the current CLUT (16-bit pixel coords)
Matrix4f            gSpriteBillboardMatrix;     // A transform matrix containing the axis vectors used for sprite billboarding
Matrix4f            gViewProjMatrix;            // The combined view and projection transform matrix for the scene
VPipelineType_Main  gOpaqueGeomPipeline;        // The pipeline to use for drawing opaque geometry
uint64_t            gClutZeroIdxMask[4];        // Bit i set: entry i of the 3D view's CLUT is '0' (a transparent texel)

//------------------------------------------------------------------------------------------------------------------------------------------
// Determine various parameters affecting the draw, including view position, projection matrix and so on
//------------------------------------------------------------------------------------------------------------------------------------------
static void RV_DetermineDrawParams() noexcept {
    // Which player's viewpoint is the world being drawn from?
    player_t& player = gPlayers[gCurPlayerIndex];
    gpViewPlayer = &player;

    // Determine the view x/y/z position and the angle in fixed point terms first.
    // If we are doing an uncapped frame-rate then use interpolation to determine these parameters.
    // Note that unlike the original renderer I am not discarding the fractional component of the player's position here.
    R_CalcLerpFactors();
    const bool bInterpolateFrame = PlayerPrefs::gbUncapFramerate;
    mobj_t& playerMobj = *player.mo;

    if (bInterpolateFrame) {
        const fixed_t newViewX = playerMobj.x;
        const fixed_t newViewY = playerMobj.y;
        const fixed_t newViewZ = player.viewz;
        const angle_t newViewAngle = playerMobj.angle;
        const fixed_t lerp = gPlayerLerpFactor;

        if (gbSnapViewZInterpolation) {
            gOldViewZ = newViewZ;
            gViewPushedZ = 0;
            gbOldViewZIsPushed = false;
            gbSnapViewZInterpolation = false;
        }

        // Note: interpolate the amount that the view was pushed by the world at a lower 15 Hz rate.
        // Sectors and enemies in the world tick at 15 Hz rather than 30 Hz like the player.
        gViewX = R_LerpCoord(gOldViewX, newViewX, lerp);
        gViewY = R_LerpCoord(gOldViewY, newViewY, lerp);

        if (gbOldViewZIsPushed) {
            // Note: when the old 'ViewZ' value already incorporates some 15 Hz (world) motion then we must remove that from
            // consideration for this 30 Hz (player) interpolation. The 15 Hz motion will be interpolated separately below.
            gViewZ = R_LerpCoord(gOldViewZ - gViewPushedZ, newViewZ - gViewPushedZ, lerp);
        } else {
            gViewZ = R_LerpCoord(gOldViewZ, newViewZ - gViewPushedZ, lerp);
        }

        if (Config::gbInterpolateSectors) {
            gViewZ += R_LerpCoord(0, gViewPushedZ, gWorldLerpFactor);   // Interpolating the 15 Hz (world) motion at a slower rate
        } else {
            gViewZ += gViewPushedZ;     // Sector interpolation is turned off, so snap motion due to sectors (the world) immediately
        }

        // The player's view angle is no longer interpolated in most situations since turning movements are now completely framerate uncapped.
        // The only exception is when the user does not have control, during demo playback and after death.
        const bool bUserControlsTurning = ((!gbDemoPlayback) && (player.playerstate == PST_LIVE));

        if (bUserControlsTurning) {
            // Normal gameplay: take into consideration how much turning movement we haven't committed to the player object yet here.
            // For net games, we must use the view angle we said we would use NEXT as that is the most up-to-date angle.
            if (gNetGame == gt_single) {
                gViewAngle = playerMobj.angle + gPlayerUncommittedTurning;
            } else {
                gViewAngle = gPlayerNextTickViewAngle + gPlayerUncommittedTurning;
            }
        } else {
            // User not currently in control: interpolate changes in player view angle
            gViewAngle = R_LerpAngle(gOldViewAngle, newViewAngle, lerp);
        }
    }
    else {
        // Originally this is all that happened
        gViewX = playerMobj.x;
        gViewY = playerMobj.y;
        gViewZ = player.viewz;
        gViewAngle = playerMobj.angle;
    }

    // PsyDoom: if the external camera is active the override the player's camera
    if (gExtCameraTicsLeft > 0) {
        gViewX = gExtCameraX;
        gViewY = gExtCameraY;
        gViewZ = gExtCameraZ;
        gViewAngle = gExtCameraAngle;
    }

    // Determine view angle sine and cosine
    gViewCos = gFineCosine[gViewAngle >> ANGLETOFINESHIFT];
    gViewSin = gFineSine[gViewAngle >> ANGLETOFINESHIFT];

    // Convert the view position and angle to float for the new renderer
    gViewXf = RV_FixedToFloat(gViewX);
    gViewYf = RV_FixedToFloat(gViewY);
    gViewZf = RV_FixedToFloat(gViewZ);
    gViewAnglef = RV_AngleToFloat(gViewAngle - ANG90);
    gViewCosf = std::cos(gViewAnglef);
    gViewSinf = std::sin(gViewAnglef);

    // Determine the view rotation matrix used for sprite billboarding
    gSpriteBillboardMatrix = Matrix4f::rotateY(-gViewAnglef);

    // Compute the view projection matrix to use
    gViewProjMatrix = VDrawing::computeTransformMatrixFor3D(gViewXf, gViewZf, gViewYf, gViewAnglef);

    // Save the VRAM coordinate of the current palette CLUT
    RV_ClutIdToClutXy(g3dViewPaletteClutId, gClutX, gClutY);

    // Which of its entries are '0', i.e a transparent texel: read from PSX VRAM, which the Vulkan renderer's copy mirrors
    {
        const Gpu::Core& psxGpu = PsxVm::gGpu;
        gClutZeroIdxMask[0] = gClutZeroIdxMask[1] = gClutZeroIdxMask[2] = gClutZeroIdxMask[3] = 0;

        for (uint32_t i = 0; i < 256; ++i) {
            const uint32_t x = ((uint32_t) gClutX + i) % psxGpu.ramPixelW;
            const uint32_t y = (uint32_t) gClutY % psxGpu.ramPixelH;

            if (psxGpu.pRam[x + (uintptr_t) y * psxGpu.ramPixelW] == 0) {
                gClutZeroIdxMask[i >> 6] |= (uint64_t) 1 << (i & 63);
            }
        }
    }

    // Determine the pipeline to use for drawing fully opaque geometry
    if (gpViewPlayer->cheats & CF_XRAYVISION) {
        gOpaqueGeomPipeline = VPipelineType_Main::World_GeomAlpha;
    } else {
        gOpaqueGeomPipeline = VPipelineType_Main::World_GeomMasked;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// The pipeline for a wall or flat that would otherwise use 'gOpaqueGeomPipeline'.
// A texture whose pixels use no '0' entry of the 3D view's CLUT never discards a fragment, so it is drawn by the pipeline without
// 'discard' (identical result): tile based GPUs then remove the hidden fragments of everything drawn underneath it (Mali forward
// pixel kill and early ZS, Adreno LRZ, PowerVR HSR), which matters with the painter's order and no depth buffer of this renderer.
//------------------------------------------------------------------------------------------------------------------------------------------
VPipelineType_Main RV_OpaqueGeomPipelineFor(texture_t& tex) noexcept {
    if (gOpaqueGeomPipeline != VPipelineType_Main::World_GeomMasked)
        return gOpaqueGeomPipeline;     // X-ray vision: everything is blended

    RV_UploadDirtyTex(tex);             // So what is in VRAM (and its CLUT indices) is current before deciding
    return (I_TexAvoidsClutIndices(tex, gClutZeroIdxMask)) ? VPipelineType_Main::World_GeomOpaque : VPipelineType_Main::World_GeomMasked;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets up shader uniforms for 3D drawing
//------------------------------------------------------------------------------------------------------------------------------------------
static void RV_SetShaderUniformsFor3D() noexcept {
    VShaderUniforms_Draw uniforms = {};
    VRenderer::initRendererUniformFields(uniforms);
    uniforms.mvpMatrix = gViewProjMatrix;

    VDrawing::setDrawUniforms(uniforms);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Renders the player's view.
// Some of the high level logic here is copied from the original renderer's 'R_RenderPlayerView'.
//------------------------------------------------------------------------------------------------------------------------------------------
void RV_RenderPlayerView() noexcept {
    // Do nothing if drawing is currently not allowed
    if (!VRenderer::isRendering())
        return;

    // Increment the marker used to determine when to update the 'draw height' for each sector
    gValidCount++;

    // Determine various draw settings and clear x-axis occlusion info to start with.
    // Then traverse the BSP tree to determine what needs to be drawn and in what order
    RV_DetermineDrawParams();
    RV_ClearOcclussion();
    RV_BuildDrawSubsecList();

    // Build the list of sprite fragments to be drawn for each subsector
    RV_BuildSpriteFragLists();

    // Init which subsectors are to have flats and sky walls drawn next.
    // We try and batch all those for performance reasons, and also to avoid visual artifacts with sprite clipping.
    RV_InitNextDrawFlats();
    RV_InitNextDrawSkyWalls();

    // Upload a new sky texture for this frame if required.
    // Also draw the base (background) sky that is needed in some scenarios.
    if (gbIsSkyVisible) {
        RV_CacheSkyTex();
        RV_DrawBackgroundSky();
    }

    // Finish UI drawing batches first (if some are still active) - so we don't disturb their current set of shader uniforms and transform matrix.
    // Then set a compatible drawing pipeline (so we can push shader constants) and then set the shader uniforms to use.
    VDrawing::endCurrentDrawBatch();
    VDrawing::setDrawPipeline(gOpaqueGeomPipeline);
    RV_SetShaderUniformsFor3D();

    // Increment the marker used to determine when to update the shading params for each sector
    gValidCount++;

    // Draw all of the subsectors back to front
    const int32_t numDrawSubsecs = (int32_t) gRvDrawSubsecs.size();

    for (int32_t drawSubsecIdx = numDrawSubsecs - 1; drawSubsecIdx >= 0; --drawSubsecIdx) {
        // Make sure the shading params for the sector are up to date
        subsector_t& subsec = *gRvDrawSubsecs[drawSubsecIdx];
        sector_t& sector = *subsec.sector;
        R_UpdateShadingParams(sector);

        // Draw all subsector sky walls, blended and masked walls
        RV_DrawSubsecSkyWalls(drawSubsecIdx);
        RV_DrawSubsecBlendedWalls(subsec);

        // Draw all subsector opaque elements and then sprites on top of that.
        // Most of the time these should all be on the same draw pipeline, so we can do batching.
        RV_DrawSubsecOpaqueWalls(subsec);
        RV_DrawSubsecFloors(drawSubsecIdx);
        RV_DrawSubsecCeilings(drawSubsecIdx);
        RV_DrawSubsecSpriteFrags(drawSubsecIdx);
    }

    // Cleanup after drawing the world: need to clear the draw order for each drawn subsector
    RV_ClearSubsecDrawIndexes();

    // Switch back to UI renderng and draw a letterbox in the horizontal region where the status bar would be
    Utils::onBeginUIDrawing(VPipelineType_Main::Colored);
    RV_DrawWidescreenStatusBarLetterbox();

    // Draw the player's weapon
    if (gExtCameraTicsLeft <= 0) {
        Utils::onBeginUIDrawing(VPipelineType_Main::UI_8bpp);
        RV_DrawWeapon();
    }
}

#endif  // #if PSYDOOM_VULKAN_RENDERER
