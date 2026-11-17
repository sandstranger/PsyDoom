#pragma once

#include "Macros.h"

#include <cstdint>

struct GammaTable;
struct SDL_Window;

BEGIN_NAMESPACE(Video)

class IVideoBackend;

// The original render/draw and output/display resolution of the game: the game rendered to a 256x240 framebuffer but stretched this image to
// approximately 292.57x240 in square pixel terms - even though the game asks for a 320x200 'pixel' display (CRTs did not have pixels).
//
// For more info on PS1, NES and PSX Doom pixel aspect ratios, see:
//  https://github.com/libretro/beetle-psx-libretro/issues/510
//  http://forums.nesdev.com/viewtopic.php?t=8983
//  https://doomwiki.org/wiki/Sony_PlayStation
//
static constexpr int32_t ORIG_DRAW_RES_X = 256;
static constexpr int32_t ORIG_DRAW_RES_Y = 240;
static constexpr int32_t ORIG_DISP_RES_X = 292;
static constexpr int32_t ORIG_DISP_RES_Y = 240;

// Which video backend is being used
enum BackendType {
    SDL,        // Using an SDL backend which only supports the classic renderer
    Vulkan,     // Using a Vulkan backend which supports both the classic renderer and the new Vulkan renderer
    GLES        // Using an OpenGL ES 3.0/2.0 backend which supports both the classic renderer and the new renderer (its GLES version)
};

extern BackendType  gBackendType;
extern SDL_Window*  gpSdlWindow;
extern int32_t      gTopOverscan;
extern int32_t      gBotOverscan;
extern GammaTable   gGammaTbl;

void initVideo() noexcept;
void shutdownVideo() noexcept;

void getClassicFramebufferWindowRect(
    const float windowW,
    const float windowH,
    float& rectX,
    float& rectY,
    float& rectW,
    float& rectH
) noexcept;

void displayFramebuffer() noexcept;
bool isUsingVulkanVideoBackend() noexcept;
bool isUsingGlesVideoBackend() noexcept;
bool isUsingNewRendererBackend() noexcept;
bool isUsingVulkanRenderPath() noexcept;
void onRenderDeviceReset() noexcept;
IVideoBackend& getCurrentBackend() noexcept;

END_NAMESPACE(Video)
