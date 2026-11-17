#pragma once

//------------------------------------------------------------------------------------------------------------------------------------------
// This header contains some basic types used by the Vulkan renderer
//------------------------------------------------------------------------------------------------------------------------------------------
#if PSYDOOM_VULKAN_RENDERER

#include "Matrix4.h"

#include <cstdint>

//------------------------------------------------------------------------------------------------------------------------------------------
// Graphics pipeline types for the Vulkan Renderer.
// Affects the primitive types expected, shaders used, blending mode and so on.
//------------------------------------------------------------------------------------------------------------------------------------------

// Main renderpath pipelines
enum class VPipelineType_Main : uint8_t {
    Lines,                      // Solid colored lines, no blending: can be in either 2D or 3D (for debug use for example)
    Colored,                    // Solid colored triangles, no blending: can be in either 2D or 3D (for debug use for example)
    UI_4bpp,                    // 2D/UI: texture mapped with clamping @ 4bpp, masked but no blending
    UI_8bpp,                    // 2D/UI: texture mapped with clamping @ 8bpp, masked but no blending
    UI_8bpp_Add,                // 2D/UI: texture mapped with clamping @ 8bpp, masked & additive blended (used for player weapon when partial invisibility is active)
    UI_16bpp,                   // 2D/UI: texture mapped with clamping @ 16bpp, masked but no blending
    World_GeomMasked,           // 3D world/view: textured with wrapping @ 8bpp and lit, masked but no blending
    World_GeomOpaque,           // 3D world/view: as 'World_GeomMasked' for textures that use no '0' CLUT entry: no 'discard' (tile based GPUs)
    World_GeomAlpha,            // 3D world/view: textured with wrapping @ 8bpp and lit, masked & alpha blended
    World_SpriteMasked,         // 3D world/view: textured with clamping @ 8bpp and lit, masked but no blending
    World_SpriteAlpha,          // 3D world/view: textured with clamping @ 8bpp and lit, masked & alpha blended
    World_SpriteAdditive,       // 3D world/view: textured with clamping @ 8bpp and lit, masked & additive blended
    World_SpriteSubtractive,    // 3D world/view: textured with clamping @ 8bpp and lit, masked & subtractive blended
    World_Sky,                  // 3D world/view: used to draw the sky, masked but no blending
    MsaaResolve,                // Simple shader that resolves MSAA samples
    GammaAdjustPostProcess,     // Simple shader that does a post-process gamma adjustment
    NUM_TYPES                   // Convenience declaration...
};

// Crossfade renderpath pipelines
enum class VPipelineType_Crossfade : uint8_t {
    Crossfade,                  // Used for doing crossfades
    CrossfadeGammaAdjusted,     // Used for doing crossfades (gamma adjusted)
    NUM_TYPES                   // Convenience declaration...
};

// Loading plaque renderpath pipelines
enum class VPipelineType_LoadingPlaque : uint8_t {
    LoadingPlaque,                  // Used for drawing loading plaques
    LoadingPlaqueGammaAdjusted,     // Used for drawing loading plaques (gamma adjusted)
    NUM_TYPES                       // Convenience declaration...
};

// PSX renderpath pipelines
enum class VPipelineType_PSX : uint8_t {
    GammaAdjustBlit,            // Used for blitting an image to an output region with gamma adjustment
    NUM_TYPES                   // Convenience declaration...
};

//------------------------------------------------------------------------------------------------------------------------------------------
// Light diminishing mode flags for 3D view triangles
//------------------------------------------------------------------------------------------------------------------------------------------
static constexpr uint8_t V_LIGHTDIM_FLAG_NO_LIGHTDIM = 0x1;     // Render with no light diminishing (used for things/sprites)
static constexpr uint8_t V_LIGHTDIM_FLAG_WALLS       = 0x2;     // Render using the wall light diminishing: a bit brighter than the floor
static constexpr uint8_t V_LIGHTDIM_FLAG_FLATS       = 0x4;     // Render using the floor light diminishing: darkens quicker than walls
static constexpr uint8_t V_LIGHTDIM_FLAG_ALPHA_0_05  = 0x8;     // Use Alpha 0.05 style light diminishing? Affects maximum brightness.

//------------------------------------------------------------------------------------------------------------------------------------------
// Shader uniforms for the various shaders in the main 'draw' subpass
//------------------------------------------------------------------------------------------------------------------------------------------
struct VShaderUniforms_Draw {
    // Modelview projection matrix: used to transform vertices in the vertex shader
    Matrix4f mvpMatrix;

    // Coordinate system conversion: scalars to convert from normalized device coords to the original PSX framebuffer coords (256x240).
    // Also where the where the original PSX framebuffer coords start in NDC space.
    float ndcToPsxScaleX, ndcToPsxScaleY;
    float psxNdcOffsetX, psxNdcOffsetY;

    // Inverse draw resolution (1 / draw resolution).
    // This is used to convert from screen space to normalized device coords.
    float invDrawResX, invDrawResY;
};

// Make sure shader uniforms are not bigger than the minimum push constant range required by the Vulkan standard, which is 128 bytes.
// We pass these as push constants because the data is very small and for a lot of GPUs push constants are more optimal.
static_assert(sizeof(VShaderUniforms_Draw) <= 128);

//------------------------------------------------------------------------------------------------------------------------------------------
// Shader uniforms for crossfading: contains just a single linear interpolation factor to control blending between the framebuffers
//------------------------------------------------------------------------------------------------------------------------------------------
struct VShaderUniforms_Crossfade {
    float lerpAmount;
};

static_assert(sizeof(VShaderUniforms_Crossfade) <= 128);    // Same restrictions apply as with 'VShaderUniforms_Draw' - see above...

//------------------------------------------------------------------------------------------------------------------------------------------
// Vulkan renderer vertex type: used for all direct drawing operations
//------------------------------------------------------------------------------------------------------------------------------------------
struct VVertex_Draw {
    // XYZ Position of the vertex
    float x, y, z;

    // Color for the vertex: rgb where '128' is regarded as 1.0
    uint8_t r, g, b;

    // Light diminishing mode flags (V_LIGHTDIM_FLAG_XXX) for the 3D view: unused by UI shaders
    uint8_t lightDimModeFlags;

    // 2D Texture coordinates for the vertex.
    // These coordinates are in terms of 16-bit pixels in VRAM.
    float u, v;

    // Texture wrapping window: x and y position
    uint16_t texWinX, texWinY;

    // Texture wrapping window: width and height
    uint16_t texWinW, texWinH;

    // CLUT location for 4-bit and 8-bit color modes: x and y
    uint16_t clutX, clutY;

    // When a pixel is flagged as 'semi-transparent', the RGBA color to multiply that pixel by.
    // Used to control blending when semi-transparency is active; a value of '128' is regarded as 1.0.
    // The 'alpha' semi transparency multiply component effectively is the alpha for the vertex.
    uint8_t stmulR, stmulG, stmulB, stmulA;
};

#if PSYDOOM_GLES_RENDERER
//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer, OpenGL ES 3.0 with instancing: one primitive (quad, triangle or line) drawn as one instance, instead of as the vertices
// of its triangles ('VVertex_Draw'). The vertex shader picks the corner by 'gl_VertexID'.
//
// The corners are in triangle strip order: a quad is drawn as a 4 vertex strip, giving exactly the same 2 triangles (and winding) as the
// 6 vertices of 'VVertex_Draw'. A triangle repeats its last corner (the second triangle of the strip has no area), a line uses the
// first 2 corners. What is the same for all the vertices of a primitive with 'VVertex_Draw' is stored once.
//------------------------------------------------------------------------------------------------------------------------------------------
struct VInstance_Draw {
    float       x[4];               // Corner positions
    float       y[4];
    float       z[4];
    float       u[4];               // Corner texture coordinates
    float       v[4];
    uint8_t     color[4][4];        // Per corner: r, g, b, light diminishing mode flags (as in 'VVertex_Draw')
    uint16_t    texWinX, texWinY;   // Texture window and CLUT, as in 'VVertex_Draw'
    uint16_t    texWinW, texWinH;
    uint16_t    clutX, clutY;
    uint8_t     stmulR, stmulG, stmulB, stmulA;
};

static_assert(sizeof(VInstance_Draw) == 112);
#endif

//------------------------------------------------------------------------------------------------------------------------------------------
// Vulkan renderer vertex type: used for MSAA resolve and just contains a 2D position in normalized device coords
//------------------------------------------------------------------------------------------------------------------------------------------
struct VVertex_MsaaResolve {
    float x, y;
};

//------------------------------------------------------------------------------------------------------------------------------------------
// Vulkan renderer vertex type: contains a simple 2D position and a UV coordinate.
// Used for crossfade drawing and also for drawing loading plaques.
//------------------------------------------------------------------------------------------------------------------------------------------
struct VVertex_XyUv {
    float x, y;
    float u, v;
};

#endif  // #if PSYDOOM_VULKAN_RENDERER
