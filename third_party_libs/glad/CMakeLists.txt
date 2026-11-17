set(SOURCE_FILES
    "include/glad/glad.h"
    "src/glad.c"
)

set(OTHER_FILES
)

add_library(${GLAD_TGT_NAME} STATIC ${SOURCE_FILES} ${OTHER_FILES})
setup_source_groups("${SOURCE_FILES}" "${OTHER_FILES}")

target_include_directories(${GLAD_TGT_NAME} PUBLIC "include")

# The loader was generated with '--omit-khrplatform': 'khrplatform.h' (the 'khronos_xxx' types) must be included before 'glad.h'.
# The Android NDK has it in the sysroot. Elsewhere use the copy that comes with SDL, which defines the same types.
# Users of the library include it themselves first (see 'game/PsyDoom/Gles/GlesApi.h').
if (ANDROID)
    set(GLAD_KHRPLATFORM_HEADER "KHR/khrplatform.h")
else()
    set(GLAD_KHRPLATFORM_HEADER "${PROJECT_SOURCE_DIR}/third_party_libs/libsdl/sdl2/include/SDL_opengles2_khrplatform.h")
endif()

if (MSVC)
    target_compile_options(${GLAD_TGT_NAME} PRIVATE "/FI${GLAD_KHRPLATFORM_HEADER}")
else()
    target_compile_options(${GLAD_TGT_NAME} PRIVATE "SHELL:-include ${GLAD_KHRPLATFORM_HEADER}")
endif()

add_common_target_compile_options(${GLAD_TGT_NAME})
