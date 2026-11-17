#pragma once

//------------------------------------------------------------------------------------------------------------------------------------------
// Includes the OpenGL ES 3.0 API for the GLES renderer: the 'glad' loader (third_party_libs/glad).
// The loader was generated without 'khrplatform.h', which provides the 'khronos_xxx' types it uses, so include that first.
// The Android NDK has it in the sysroot, SDL comes with a copy of it for the other platforms.
//------------------------------------------------------------------------------------------------------------------------------------------
#if PSYDOOM_GLES_RENDERER

#if defined(ANDROID) || __has_include(<KHR/khrplatform.h>)
    #include <KHR/khrplatform.h>
#else
    #include <SDL_opengles2_khrplatform.h>
#endif

#include <glad/glad.h>

#endif  // #if PSYDOOM_GLES_RENDERER
