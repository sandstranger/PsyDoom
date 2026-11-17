#pragma once

#if PSYDOOM_GLES_RENDERER

#include "GlesApi.h"
#include "Macros.h"

#include <cstdint>

//------------------------------------------------------------------------------------------------------------------------------------------
// The shader programs of the GLES renderer: GLSL ES 3.00 versions of the Vulkan renderer's shaders ('vulkan_shaders'), with what
// were specialization constants there turned into preprocessor defines (one program per variant).
// Programs come from 'GShaderCache' (RAM, then disk) when possible, otherwise they are compiled and their binaries cached.
//------------------------------------------------------------------------------------------------------------------------------------------
enum class GProgram : uint8_t {
    Colored,                // Solid colored lines and triangles
    UI_4bpp,                // UI sprites, 4bpp textures
    UI_8bpp,                // UI sprites, 8bpp textures
    UI_16bpp,               // UI sprites, 16bpp textures
    World_Wrap,             // 3D world walls and flats: wrapped textures, '0' texels discarded
    World_WrapOpaque,       // 3D world walls and flats whose texture uses no '0' CLUT entry: no 'discard' (tile based GPU hidden surface removal)
    World_Clamp,            // 3D world sprites: clamped textures, '0' texels discarded
    Sky,                    // The sky
    Present,                // Copies the frame drawn by the main render path to the screen
    Present_Gamma,          // As 'Present' with gamma adjustment
    PsxPresent,             // Shows the PSX framebuffer (classic renderer) on the screen, decoding the PSX 16-bit pixels
    PsxPresent_Gamma,       // As 'PsxPresent' with gamma adjustment
    Crossfade,              // Crossfades between two frames drawn by the main render path
    Crossfade_Gamma,        // As 'Crossfade' with gamma adjustment
    Quad,                   // A textured quad: external surfaces (movies, logos) and loading plaque backgrounds
    Quad_Gamma,             // As 'Quad' with gamma adjustment
    QuadMasked,             // A textured quad whose fully transparent texels are discarded: loading plaques
    QuadMasked_Gamma,       // As 'QuadMasked' with gamma adjustment
    NUM_PROGRAMS
};

// Uniform locations and such of a program (-1 for what a program doesn't have)
struct GProgramInfo {
    GLuint  program;
    GLint   uPosRect;       // Quad programs: the corners of the quad in NDC (left x, top y, right x, bottom y)
    GLint   uUvRect;        // Quad programs: the texture coordinates of the corners (left u, top v, right u, bottom v)
    GLint   uLerp;          // Crossfade programs: how far into the crossfade
};

BEGIN_NAMESPACE(GShaders)

// The binding point of the uniform block of the draw programs ('VShaderUniforms_Draw')
static constexpr GLuint DRAW_UNIFORMS_BINDING = 0;

// Texture units used by the programs
static constexpr GLuint TEX_UNIT_VRAM       = 0;    // Draw programs: PSX VRAM (R16UI)
static constexpr GLuint TEX_UNIT_SRC        = 1;    // Post programs: the image drawn (crossfade: the image faded from)
static constexpr GLuint TEX_UNIT_SRC2       = 2;    // Crossfade: the image faded to
static constexpr GLuint TEX_UNIT_GAMMA_LUT  = 3;    // Post programs: the 256x1 gamma remap table

void init(const bool bUse16BitShading, const bool bUseProgramBinaries) noexcept;
void destroy(const bool bContextLost) noexcept;
const GProgramInfo& get(const GProgram program) noexcept;

END_NAMESPACE(GShaders)

#endif  // #if PSYDOOM_GLES_RENDERER
