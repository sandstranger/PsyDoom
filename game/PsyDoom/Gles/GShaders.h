#pragma once

#if PSYDOOM_GLES_RENDERER

#include "GlesApi.h"
#include "Macros.h"

#include <cstdint>

//------------------------------------------------------------------------------------------------------------------------------------------
// The shader programs of the GLES renderer: versions of the Vulkan renderer's shaders ('vulkan_shaders') for both of its paths:
//
//  - OpenGL ES 3.0: GLSL ES 3.00, PSX VRAM as an R16UI texture read with 'texelFetch' and integer operations, as with Vulkan.
//  - OpenGL ES 2.0: GLSL ES 1.00, which has no integer textures or operations: PSX VRAM is a 2 bytes per texel texture (RG8 or
//    luminance alpha) and the 16-bit texels and CLUT indexes are decoded with float arithmetic, exact for these small integers.
//    The draw programs need 'highp' floats in fragment shaders for that (VRAM coordinates go up to 8192).
//
// What were specialization constants with Vulkan are preprocessor defines here (one program per variant).
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

static constexpr uint32_t NUM_DRAW_PROGRAMS = (uint32_t) GProgram::Sky + 1;     // The draw programs come first

// Which OpenGL ES path the renderer uses
enum class GApi : uint8_t {
    Gles2,
    Gles3
};

// Uniform locations and such of a program (-1 for what a program doesn't have)
struct GProgramInfo {
    GLuint  program;
    GLint   uPosRect;           // Post programs: the corners of the quad drawn in NDC (left x, top y, right x, bottom y)
    GLint   uUvRect;            // Post programs: the texture coordinates of the corners (left u, top v, right u, bottom v)
    GLint   uLerp;              // Crossfade programs: how far into the crossfade
    GLint   uMvpMatrix;         // OpenGL ES 2.0 draw programs (no uniform blocks): the fields of 'VShaderUniforms_Draw'...
    GLint   uNdcToPsxScale;
    GLint   uPsxNdcOffset;
    GLint   uInvDrawRes;
    GLint   uVramInvSize;       // OpenGL ES 2.0 draw programs: 1 / the size of PSX VRAM (texture coordinates are normalized)
};

// Options for making the programs
struct GShaderOptions {
    GApi    api;
    bool    bUse16BitShading;       // Shade with the PSX's 16-bit precision
    bool    bUseProgramBinaries;    // Use the program binary cache (needs a driver with program binary formats)
    bool    bHighpFragment;         // OpenGL ES 2.0: 'highp' floats in fragment shaders (needed by the draw programs)
    bool    bMakeDrawPrograms;      // Make the draw programs (not if the main render path is not supported)
    bool    bInstancing;            // OpenGL ES 3.0: the draw programs take one instance per primitive ('VInstance_Draw')
    char    vramHiChannel;          // OpenGL ES 2.0: the channel with the high byte of the VRAM texels ('g' for RG8, 'a' for luminance alpha)
};

BEGIN_NAMESPACE(GShaders)

// The binding point of the uniform block of the draw programs (OpenGL ES 3.0: 'VShaderUniforms_Draw')
static constexpr GLuint DRAW_UNIFORMS_BINDING = 0;

// Vertex attribute locations: draw programs ('VVertex_Draw') and post programs (quad corners)
static constexpr GLuint ATTRIB_POS              = 0;
static constexpr GLuint ATTRIB_COLOR_LIGHTDIM   = 1;
static constexpr GLuint ATTRIB_UV               = 2;
static constexpr GLuint ATTRIB_TEXWIN_POS       = 3;
static constexpr GLuint ATTRIB_TEXWIN_SIZE      = 4;
static constexpr GLuint ATTRIB_CLUT_POS         = 5;
static constexpr GLuint ATTRIB_STMUL            = 6;
static constexpr GLuint NUM_DRAW_ATTRIBS        = 7;
static constexpr GLuint ATTRIB_CORNER           = 0;

// Vertex attribute locations of the draw programs with instancing (OpenGL ES 3.0): 'VInstance_Draw', one per instance
static constexpr GLuint INST_ATTRIB_X           = 0;
static constexpr GLuint INST_ATTRIB_Y           = 1;
static constexpr GLuint INST_ATTRIB_Z           = 2;
static constexpr GLuint INST_ATTRIB_U           = 3;
static constexpr GLuint INST_ATTRIB_V           = 4;
static constexpr GLuint INST_ATTRIB_COLORS      = 5;
static constexpr GLuint INST_ATTRIB_TEXWIN      = 6;
static constexpr GLuint INST_ATTRIB_CLUT_STMUL  = 7;
static constexpr GLuint NUM_INST_ATTRIBS        = 8;

// Texture units used by the programs
static constexpr GLuint TEX_UNIT_VRAM       = 0;    // Draw programs: PSX VRAM
static constexpr GLuint TEX_UNIT_SRC        = 1;    // Post programs: the image drawn (crossfade: the image faded from)
static constexpr GLuint TEX_UNIT_SRC2       = 2;    // Crossfade: the image faded to
static constexpr GLuint TEX_UNIT_GAMMA_LUT  = 3;    // Post programs: the 256x1 gamma remap table

void init(const GShaderOptions& options) noexcept;
void destroy(const bool bContextLost) noexcept;
void setVramSize(const uint32_t vramW, const uint32_t vramH) noexcept;
const GProgramInfo& get(const GProgram program) noexcept;

END_NAMESPACE(GShaders)

#endif  // #if PSYDOOM_GLES_RENDERER
