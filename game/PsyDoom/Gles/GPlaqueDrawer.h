#pragma once

#if PSYDOOM_GLES_RENDERER

#include "Macros.h"

#include <cstdint>

struct texture_t;

//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: draws loading plaques over the last frame of the main render path (the same as 'VPlaqueDrawer' of the Vulkan renderer)
//------------------------------------------------------------------------------------------------------------------------------------------
BEGIN_NAMESPACE(GPlaqueDrawer)

void drawPlaque(texture_t& plaqueTex, const int16_t plaqueX, const int16_t plaqueY, const int16_t clutId) noexcept;

END_NAMESPACE(GPlaqueDrawer)

#endif  // #if PSYDOOM_GLES_RENDERER
