#pragma once

#if PSYDOOM_GLES_RENDERER

#include "Macros.h"

#include <cstdint>

//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: crossfades between the last two frames drawn by the main render path (the same as 'VCrossfader' of the Vulkan renderer)
//------------------------------------------------------------------------------------------------------------------------------------------
BEGIN_NAMESPACE(GCrossfader)

void doPreCrossfadeSetup() noexcept;
void doCrossfade(const int32_t vblanksDuration) noexcept;

END_NAMESPACE(GCrossfader)

#endif  // #if PSYDOOM_GLES_RENDERER
