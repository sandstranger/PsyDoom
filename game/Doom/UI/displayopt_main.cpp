//------------------------------------------------------------------------------------------------------------------------------------------
// This is an entirely new menu added for PsyDoom: it provides extra graphics/display related options.
// It is not available in multiplayer, similar to other nested menus in the options screen.
//------------------------------------------------------------------------------------------------------------------------------------------
#if PSYDOOM_MODS

#include "displayopt_main.h"

#include "Doom/Base/i_main.h"
#include "Doom/Base/i_misc.h"
#include "Doom/Base/s_sound.h"
#include "Doom/Base/sounds.h"
#include "Doom/d_main.h"
#include "Doom/Game/g_game.h"
#include "Doom/Game/p_tick.h"
#include "Doom/Renderer/r_data.h"
#include "m_main.h"
#include "o_main.h"
#include "PsyDoom/Game.h"
#include "PsyDoom/GammaTable.h"
#include "PsyDoom/PlayerPrefs.h"
#include "PsyDoom/Utils.h"
#include "PsyDoom/Video.h"
#include "PsyDoom/Vulkan/VRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// The available menu items
enum MenuItem : int32_t {
    menu_gamma_adjust,
    menu_stat_display,
    menu_uncapped_framerate,
#if PSYDOOM_VULKAN_RENDERER
    menu_renderer,
#endif
    menu_exit,
    num_menu_items
};

//------------------------------------------------------------------------------------------------------------------------------------------
// Draw the cursor at the specified position
//------------------------------------------------------------------------------------------------------------------------------------------
static void DrawCursor(const int16_t cursorX, const int16_t cursorY) noexcept {
    const uint8_t skullTexV = (Game::gGameType != GameType::Doom_Alpha_0_05) ? M_SKULL_TEX_V : M_SKULL_TEX_V_ALPHA_0_05;

    I_DrawSprite(
        gTex_STATUS.texPageId,
        Game::getTexClut_STATUS(),
        (int16_t) cursorX - 24,
        (int16_t) cursorY - 2,
        (int16_t)(gTex_STATUS.texPageCoordX + M_SKULL_TEX_U + (uint8_t) gCursorFrame * M_SKULL_W),
        (int16_t)(gTex_STATUS.texPageCoordY + skullTexV),
        M_SKULL_W,
        M_SKULL_H
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the menu
//------------------------------------------------------------------------------------------------------------------------------------------
void DisplayOpt_Init() noexcept {
    S_StartSound(nullptr, sfx_pistol);

    // Initialize cursor position and vblanks until move
    gCursorFrame = 0;
    gCursorPos[gCurPlayerIndex] = 0;
    gVBlanksUntilMenuMove[gCurPlayerIndex] = 0;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Shuts down the menu
//------------------------------------------------------------------------------------------------------------------------------------------
void DisplayOpt_Shutdown([[maybe_unused]] const gameaction_t exitAction) noexcept {
    gCursorPos[gCurPlayerIndex] = 0;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Runs update logic for the menu: does menu controls
//------------------------------------------------------------------------------------------------------------------------------------------
gameaction_t DisplayOpt_Update() noexcept {
    // PsyDoom: in all UIs tick only if vblanks are registered as elapsed; this restricts the code to ticking at 30 Hz for NTSC
    const uint32_t playerIdx = gCurPlayerIndex;

    if (gPlayersElapsedVBlanks[playerIdx] <= 0) {
        gbKeepInputEvents = true;   // Don't consume 'key pressed' etc. events yet, not ticking...
        return ga_nothing;
    }

    // Animate the skull cursor
    if ((gGameTic > gPrevGameTic) && ((gGameTic & 3) == 0)) {
        gCursorFrame ^= 1;
    }

    // Gather menu inputs and exit if the back button has just been pressed
    const TickInputs& inputs = gTickInputs[playerIdx];
    const TickInputs& oldInputs = gOldTickInputs[playerIdx];

    const bool bMenuBack = (inputs.fMenuBack() && (!oldInputs.fMenuBack()));
    const bool bMenuOk = (inputs.fMenuOk() && (!oldInputs.fMenuOk()));
    const bool bMenuUp = inputs.fMenuUp();
    const bool bMenuDown = inputs.fMenuDown();
    const bool bMenuLeft = inputs.fMenuLeft();
    const bool bMenuRight = inputs.fMenuRight();
    const bool bMenuMove = (bMenuUp || bMenuDown || bMenuLeft || bMenuRight);

    if (bMenuBack) {
        S_StartSound(nullptr, sfx_pistol);
        return ga_exit;
    }

    // Check for up/down movement
    if (!bMenuMove) {
        // If there are no direction buttons pressed then the next move is allowed instantly
        gVBlanksUntilMenuMove[playerIdx] = 0;
    } else {
        // Direction buttons pressed or held down, check to see if we can move up/down now
        gVBlanksUntilMenuMove[playerIdx] -= gPlayersElapsedVBlanks[playerIdx];

        if (gVBlanksUntilMenuMove[playerIdx] <= 0) {
            gVBlanksUntilMenuMove[playerIdx] = 15;

            if (bMenuDown) {
                gCursorPos[playerIdx]++;

                if (gCursorPos[playerIdx] >= num_menu_items) {
                    gCursorPos[playerIdx] = 0;
                }

                S_StartSound(nullptr, sfx_pstop);
            }
            else if (bMenuUp) {
                gCursorPos[playerIdx]--;

                if (gCursorPos[playerIdx] < 0) {
                    gCursorPos[playerIdx] = num_menu_items - 1;
                }

                S_StartSound(nullptr, sfx_pstop);
            }
        }
    }

    // Handle option actions and adjustment
    switch ((MenuItem) gCursorPos[playerIdx]) {
        // Gamma adjustments
        case menu_gamma_adjust: {
            if (bMenuLeft || bMenuRight) {
                const int32_t oldGamma1000 = PlayerPrefs::gGamma1000;

                // Use a logarithmic scale to move the slider evenly (perceptually) in both the positive and negative directions
                const float logGammaMin = std::log((float) PlayerPrefs::GAMMA_MIN / 1000.0f);
                const float logGammaMax = std::log((float) PlayerPrefs::GAMMA_MAX / 1000.0f);
                const float logGammaRange = logGammaMax - logGammaMin;
                const float logGammaStep = logGammaRange / 800.0f; // Allow 800 slider steps
                const float logOldGamma = std::log((float) oldGamma1000 / 1000.0f);

                const int32_t gammaReduceStep = std::min((int32_t)(std::exp(logOldGamma - logGammaStep) * 1000.0f + 0.5f) - oldGamma1000, -1);
                const int32_t gammaIncreaseStep = std::max((int32_t)(std::exp(logOldGamma + logGammaStep) * 1000.0f + 0.5f) - oldGamma1000, +1);
                
                // Helper: called when the gamma setting has changed: rebuilds gamma adjustment LUTs.
                const auto onGammaSettingChanged = []() noexcept {
                    Video::gGammaTbl.build((float) PlayerPrefs::gGamma1000 / 1000.0f);
                    
                    #if PSYDOOM_VULKAN_RENDERER
                        if (Video::isUsingNewRendererBackend()) {
                            VRenderer::rebuildGammaAdjustTex();     // Note: dispatched to the GLES renderer for the GLES backend
                        }
                    #endif
                };

                // Note: when crossing the 1.0 gamma threshold snap to it
                if (bMenuRight) {
                    int32_t newGamma1000 = std::min(oldGamma1000 + gammaIncreaseStep, PlayerPrefs::GAMMA_MAX);
                    const bool bSnapToDefaultGamma = ((oldGamma1000 < PlayerPrefs::GAMMA_DEFAULT) && (newGamma1000 > PlayerPrefs::GAMMA_DEFAULT));
                    newGamma1000 = (bSnapToDefaultGamma) ? PlayerPrefs::GAMMA_DEFAULT : newGamma1000;
                    PlayerPrefs::gGamma1000 = newGamma1000;

                    if (oldGamma1000 != newGamma1000) {
                        if ((newGamma1000 / 10) & 1) {
                            S_StartSound(nullptr, sfx_stnmov);
                        }

                        onGammaSettingChanged();
                    }
                }
                else {
                    int32_t newGamma1000 = std::max(oldGamma1000 + gammaReduceStep, PlayerPrefs::GAMMA_MIN);
                    const bool bSnapToDefaultGamma = ((oldGamma1000 < PlayerPrefs::GAMMA_DEFAULT) && (newGamma1000 > PlayerPrefs::GAMMA_DEFAULT));
                    newGamma1000 = (bSnapToDefaultGamma) ? PlayerPrefs::GAMMA_DEFAULT : newGamma1000;
                    PlayerPrefs::gGamma1000 = newGamma1000;

                    if (oldGamma1000 != newGamma1000) {
                        if ((newGamma1000 / 10) & 1) {
                            S_StartSound(nullptr, sfx_stnmov);
                        }

                        onGammaSettingChanged();
                    }
                }
            }
        }   break;

        // Stat display setting
        case menu_stat_display: {
            if (bMenuLeft && (!oldInputs.fMenuLeft()) && (PlayerPrefs::gStatDisplayMode > StatDisplayMode::None)) {
                PlayerPrefs::gStatDisplayMode = (StatDisplayMode)((int32_t) PlayerPrefs::gStatDisplayMode - 1);
                S_StartSound(nullptr, sfx_swtchx);
            }
            else if (bMenuRight && (!oldInputs.fMenuRight()) && (PlayerPrefs::gStatDisplayMode < StatDisplayMode::MapOnly_KillsSecretsAndItems)) {
                PlayerPrefs::gStatDisplayMode = (StatDisplayMode)((int32_t) PlayerPrefs::gStatDisplayMode + 1);
                S_StartSound(nullptr, sfx_swtchx);
            }
        }   break;

        // Turn on/off uncapped framerate
        case menu_uncapped_framerate: {
            if (bMenuLeft && PlayerPrefs::gbUncapFramerate) {
                PlayerPrefs::gbUncapFramerate = false;
                S_StartSound(nullptr, sfx_swtchx);
            }
            else if (bMenuRight && (!PlayerPrefs::gbUncapFramerate)) {
                PlayerPrefs::gbUncapFramerate = true;
                S_StartSound(nullptr, sfx_swtchx);
            }
        }   break;

    #if PSYDOOM_VULKAN_RENDERER
        // Renderer toggle
        case menu_renderer: {
            const bool bCanSwitchRenderers = Video::isUsingNewRendererBackend();

            if (bCanSwitchRenderers) {
                if (bMenuLeft && (!Video::isUsingVulkanRenderPath())) {
                    VRenderer::switchToMainVulkanRenderPath();
                    S_StartSound(nullptr, sfx_swtchx);
                }
                else if (bMenuRight && Video::isUsingVulkanRenderPath()) {
                    VRenderer::switchToPsxRenderPath();
                    S_StartSound(nullptr, sfx_swtchx);
                }
            }

            // If renderer switch is not possible and an attempt was made to do so then play this sound
            if (!bCanSwitchRenderers) {
                if ((bMenuLeft && (!oldInputs.fMenuLeft())) || (bMenuRight && (!oldInputs.fMenuRight()))) {
                    S_StartSound(nullptr, sfx_itemup);
                }
            }
        }   break;
    #endif  // #if PSYDOOM_VULKAN_RENDERER

        // Exit to the options menu
        case menu_exit: {
            if (bMenuOk) {
                S_StartSound(nullptr, sfx_pistol);
                return ga_exit;
            }
        } break;

        default:
            break;
    }

    return ga_nothing;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Draws the menu
//------------------------------------------------------------------------------------------------------------------------------------------
void DisplayOpt_Draw() noexcept {
    // Increment the frame count for the texture cache and draw the background
    I_IncDrawnFrameCount();
    Utils::onBeginUIDrawing();
    O_DrawBackground(gTex_OptionsBg, Game::getTexClut_OptionsBg(), 128, 128, 128);

    // Don't do any rendering if we are about to exit the menu
    if (gGameAction == ga_nothing) {
        // Menu title
        I_DrawString(-1, 20, "Display Settings");

        // Draw the gamma adjustment slider
        int16_t cursorX = 62;
        int16_t cursorY = 50;

        {
            // Draw the label for the slider
            const int16_t menuItemX = 62;
            const int16_t menuItemY = 50;

            const int32_t gamma = PlayerPrefs::gGamma1000;
            char gammaAdjustLabel[32];

            std::snprintf(
                gammaAdjustLabel,
                sizeof(gammaAdjustLabel),
                "Gamma %d.%03d",
                gamma / 1000,
                gamma % 1000
            );

            I_DrawString(menuItemX, menuItemY, gammaAdjustLabel);

            // Draw the slider background
            const int16_t sliderBgU = 0;
            const int16_t sliderBgV = (Game::gGameType != GameType::Doom_Alpha_0_05) ? 184 : 185;
            
            I_DrawSprite(
                gTex_STATUS.texPageId,
                Game::getTexClut_STATUS(),
                (int16_t)(menuItemX + 13),
                (int16_t)(menuItemY + 20),
                (int16_t)(gTex_STATUS.texPageCoordX + sliderBgU),
                (int16_t)(gTex_STATUS.texPageCoordY + sliderBgV),
                108,
                11
            );

            // Draw the slider handle.
            // Note: use a logarithmic scale to distribute the slider evenly here. 1.0 to 0.5 has much more impact than 1.0 to 1.5 for example!
            const int16_t sliderHandleU = 108;
            const int16_t sliderHandleV = (Game::gGameType != GameType::Doom_Alpha_0_05) ? 184 : 185;
            
            const float logGammaMin = std::log((float) PlayerPrefs::GAMMA_MIN / 1000.0f);
            const float logGammaMax = std::log((float) PlayerPrefs::GAMMA_MAX / 1000.0f);
            const float logGammaRange = logGammaMax - logGammaMin;
            const float logGamma = std::log((float) gamma / 1000.0f);

            const int16_t sliderVal = (int16_t)(((logGamma - logGammaMin) / logGammaRange) * 100.0f + 0.5f);

            I_DrawSprite(
                gTex_STATUS.texPageId,
                Game::getTexClut_STATUS(),
                (int16_t)(menuItemX + 14 + sliderVal),
                (int16_t)(menuItemY + 20),
                (int16_t)(gTex_STATUS.texPageCoordX + sliderHandleU),
                (int16_t)(gTex_STATUS.texPageCoordY + sliderHandleV),
                6,
                11
            );
        }

        // Draw the stats display option
        const char* statDisplayStr = "Stat Display Off";

        if (PlayerPrefs::gStatDisplayMode >= StatDisplayMode::MapOnly_KillsSecretsAndItems) {
            statDisplayStr = "Automap Only KSI";
        } else if (PlayerPrefs::gStatDisplayMode >= StatDisplayMode::MapOnly_KillsAndSecrets) {
            statDisplayStr = "Automap Only KS";
        } else if (PlayerPrefs::gStatDisplayMode >= StatDisplayMode::MapOnly_Kills) {
            statDisplayStr = "Automap Only K";
        } else if (PlayerPrefs::gStatDisplayMode >= StatDisplayMode::KillsSecretsAndItems) {
            statDisplayStr = "Stat Display KSI";
        } else if (PlayerPrefs::gStatDisplayMode >= StatDisplayMode::KillsAndSecrets) {
            statDisplayStr = "Stat Display KS";
        } else if (PlayerPrefs::gStatDisplayMode >= StatDisplayMode::Kills) {
            statDisplayStr = "Stat Display K";
        }

        I_DrawString(62, 90, statDisplayStr);

        if (gCursorPos[gCurPlayerIndex] == menu_stat_display) {
            cursorY = 90;
        }

        // Draw the uncapped framerate option
        I_DrawString(62, 112, (PlayerPrefs::gbUncapFramerate) ? "Uncapped FPS" : "Original FPS");

        if (gCursorPos[gCurPlayerIndex] == menu_uncapped_framerate) {
            cursorY = 112;
        }

        #if PSYDOOM_VULKAN_RENDERER
            // Draw the renderer option
            const bool bIsUsingVulkan = Video::isUsingVulkanRenderPath();
            const char* const newRendererName = (Video::isUsingGlesVideoBackend()) ? "GLES Renderer" : "Vulkan Renderer";
            I_DrawString(62, 134, (bIsUsingVulkan) ? newRendererName : "Classic Renderer");

            if (gCursorPos[gCurPlayerIndex] == menu_renderer) {
                cursorY = 134;
            }
        #endif

        // Draw the exit option
        I_DrawString(62, 210, "Back");

        if (gCursorPos[gCurPlayerIndex] == menu_exit) {
            cursorY = 210;
        }

        // Draw the skull cursor
        DrawCursor(cursorX, cursorY);
    }

    // PsyDoom: draw any enabled performance counters
    I_DrawEnabledPerfCounters();

    // Finish up the frame
    I_SubmitGpuCmds();
    I_DrawPresent();
}

#endif  // #if PSYDOOM_MODS
