//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: the shader programs, for the OpenGL ES 3.0 and 2.0 paths.
// Versions of the shaders in 'vulkan_shaders', which they must be kept in step with. Differences:
//
//  (1) Specialization constants become preprocessor defines: one program per variant.
//  (2) The world and sky vertex shaders remap clip space depth from Vulkan's 0..w to OpenGL's -w..w, so that geometry is clipped at
//      the near and far planes exactly as with Vulkan (the light diminishing still uses the Vulkan depth, as there).
//  (3) The post shaders (present, crossfade, quads) draw straight to the window with OpenGL's y up NDC, while the images drawn by the
//      main render path keep the Vulkan layout: first row at the top (so 'gl_FragCoord' in the sky shader matches Vulkan's).
//  (4) GLSL ES 3.00: no implicit int/uint/float conversions, casts and 'u'/'.0' literal suffixes where needed.
//  (5) GLSL ES 1.00: no integers, integer textures or flat varyings. PSX VRAM texels are read as their 2 bytes and decoded with float
//      arithmetic (exact: small integers), values that are the same at all vertices are rounded back after interpolation.
//------------------------------------------------------------------------------------------------------------------------------------------
#include "GShaders.h"

#if PSYDOOM_GLES_RENDERER

#include "Asserts.h"
#include "FatalErrors.h"
#include "GShaderCache.h"

#include <algorithm>
#include <SDL_log.h>
#include <string>
#include <vector>

BEGIN_NAMESPACE(GShaders)

//==========================================================================================================================================
// OpenGL ES 3.0: GLSL ES 3.00
//==========================================================================================================================================
static constexpr const char* const SRC3_HEADER = R"(#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
)";

// Uniforms of the draw shaders: the same as the push constants of the Vulkan renderer ('VShaderUniforms_Draw', std140 layout)
static constexpr const char* const SRC3_DRAW_UNIFORMS = R"(
layout(std140) uniform DrawUniforms {
    mat4    mvpMatrix;
    vec2    ndcToPsxScale;
    vec2    psxNdcOffset;
    vec2    invDrawRes;
} uniforms;
)";

// Vertex inputs of the draw shaders: 'VVertex_Draw'
static constexpr const char* const SRC3_DRAW_VS_INPUTS = R"(
layout(location = 0) in vec3    in_pos;
layout(location = 1) in uvec4   in_color_lightDimFlags;
layout(location = 2) in vec2    in_uv;
layout(location = 3) in uvec2   in_texWinPos;
layout(location = 4) in uvec2   in_texWinSize;
layout(location = 5) in uvec2   in_clutPos;
layout(location = 6) in uvec4   in_stmul;

// Vulkan clip space depth is 0 to w, OpenGL's is -w to w: remap so that geometry is clipped by the near and far planes as with Vulkan
void toGlClipSpaceDepth() {
    gl_Position.z = gl_Position.z * 2.0 - gl_Position.w;
}
)";

// Common fragment shader functions ('ShaderCommon_Frag.h'): reading texels from PSX VRAM
static constexpr const char* const SRC3_DRAW_FS_COMMON = R"(
uniform highp usampler2D vramTex;   // A texture containing the entirety of PSX VRAM

layout(location = 0) out vec4 out_color;

vec4 sampleVramTexel(ivec2 uv, vec4 stmul, bool bAllowDiscard) {
    uint texelBits = texelFetch(vramTex, uv, 0).r;

    if (bAllowDiscard && (texelBits == 0u))
        discard;

    vec4 color = vec4(
        float(texelBits & 0x1Fu),
        float((texelBits >> 5u) & 0x1Fu),
        float((texelBits >> 10u) & 0x1Fu),
        31.0
    ) / 31.0;

    if ((texelBits & 0x8000u) != 0u) {
        color *= stmul;
    }

    return color;
}

vec4 tex16bpp(ivec2 uv, ivec2 texWinPos, ivec2 texWinSize, vec4 stmul, bool bWrap, bool bAllowDiscard) {
    if (bWrap) {
        uv = (uv & 0xFFFFFF) % texWinSize;
    } else {
        uv = clamp(uv, ivec2(0, 0), texWinSize - 1);
    }

    uv += texWinPos;
    return sampleVramTexel(uv, stmul, bAllowDiscard);
}

vec4 tex8bpp(ivec2 uv, ivec2 texWinPos, ivec2 texWinSize, ivec2 clutPos, vec4 stmul, bool bWrap, bool bAllowDiscard) {
    if (bWrap) {
        uv = (uv & 0xFFFF) % texWinSize;
    } else {
        uv = clamp(uv, ivec2(0, 0), texWinSize - 1);
    }

    uv += texWinPos;

    uint vramTexel = texelFetch(vramTex, ivec2(uv.x / 2, uv.y), 0).r;
    uint clutIdx = (vramTexel >> (uint(uv.x & 1) * 8u)) & 0xFFu;
    return sampleVramTexel(ivec2(clutPos.x + int(clutIdx), clutPos.y), stmul, bAllowDiscard);
}

vec4 tex4bpp(ivec2 uv, ivec2 texWinPos, ivec2 texWinSize, ivec2 clutPos, vec4 stmul, bool bWrap, bool bAllowDiscard) {
    if (bWrap) {
        uv = (uv & 0xFFFF) % texWinSize;
    } else {
        uv = clamp(uv, ivec2(0, 0), texWinSize - 1);
    }

    uv += texWinPos;

    uint vramTexel = texelFetch(vramTex, ivec2(uv.x / 4, uv.y), 0).r;
    uint clutIdx = (vramTexel >> (uint(uv.x & 3) * 4u)) & 0xFu;
    return sampleVramTexel(ivec2(clutPos.x + int(clutIdx), clutPos.y), stmul, bAllowDiscard);
}

vec4 psxR5G5B5BitCrush(vec4 color) {
    return trunc(color * 31.0 + 0.0001) / 31.0;
}
)";

// colored.vert, colored.frag
static constexpr const char* const SRC3_COLORED_VS = R"(
out vec4 v_color;

void main() {
    gl_Position = uniforms.mvpMatrix * vec4(in_pos, 1.0);
    v_color = vec4(vec3(in_color_lightDimFlags.rgb), float(in_stmul.a)) / 255.0;
}
)";

static constexpr const char* const SRC3_COLORED_FS = R"(
in vec4 v_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = v_color;
}
)";

// ui.vert, ui_4bpp.frag, ui_8bpp.frag, ui_16bpp.frag ('TEX_BPP')
static constexpr const char* const SRC3_UI_VS = R"(
out vec3 v_color;
out vec2 v_uv;
flat out ivec2 v_texWinPos;
flat out ivec2 v_texWinSize;
flat out ivec2 v_clutPos;
flat out vec4 v_stmul;

void main() {
    gl_Position = uniforms.mvpMatrix * vec4(in_pos, 1.0);
    v_color = vec3(in_color_lightDimFlags.rgb) / 128.0;
    v_uv = in_uv;
    v_texWinPos = ivec2(in_texWinPos);
    v_texWinSize = ivec2(in_texWinSize);
    v_clutPos = ivec2(in_clutPos);
    v_stmul = vec4(in_stmul) / 128.0;
}
)";

static constexpr const char* const SRC3_UI_FS = R"(
in vec3 v_color;
in vec2 v_uv;
flat in ivec2 v_texWinPos;
flat in ivec2 v_texWinSize;
flat in ivec2 v_clutPos;
flat in vec4 v_stmul;

void main() {
#if TEX_BPP == 4
    out_color = tex4bpp(ivec2(floor(v_uv)), v_texWinPos, v_texWinSize, v_clutPos, v_stmul, false, true);
#elif TEX_BPP == 8
    out_color = tex8bpp(ivec2(floor(v_uv)), v_texWinPos, v_texWinSize, v_clutPos, v_stmul, false, true);
#else
    out_color = tex16bpp(ivec2(floor(v_uv)), v_texWinPos, v_texWinSize, v_stmul, false, true);
#endif

    out_color.rgb *= v_color;

#if USE_PSX_16_BIT_SHADING
    out_color = psxR5G5B5BitCrush(out_color);
#endif
}
)";

// world.vert, world.frag ('WRAP_TEXTURE', 'ALLOW_DISCARD')
static constexpr const char* const SRC3_WORLD_VS = R"(
out vec3 v_color;
out vec3 v_uv_z;
flat out vec4 v_lightDimMode;
flat out ivec2 v_texWinPos;
flat out ivec2 v_texWinSize;
flat out ivec2 v_clutPos;
flat out vec4 v_stmul;

void main() {
    gl_Position = uniforms.mvpMatrix * vec4(in_pos, 1.0);

    uint lightDimFlags = in_color_lightDimFlags.a;
    v_color = vec3(in_color_lightDimFlags.rgb);
    v_uv_z = vec3(in_uv, gl_Position.z);
    v_lightDimMode.x = ((lightDimFlags & 1u) != 0u) ? 1.0 : 0.0;        // No light diminishing
    v_lightDimMode.y = ((lightDimFlags & 2u) != 0u) ? 1.0 : 0.0;        // Wall light diminishing
    v_lightDimMode.z = ((lightDimFlags & 4u) != 0u) ? 1.0 : 0.0;        // Floor light diminishing
    v_lightDimMode.w = ((lightDimFlags & 8u) != 0u) ? 192.0 : 160.0;    // Max light diminishing intensity (stronger for Alpha 0.05)
    v_texWinPos = ivec2(in_texWinPos);
    v_texWinSize = ivec2(in_texWinSize);
    v_clutPos = ivec2(in_clutPos);
    v_stmul = vec4(in_stmul) / 128.0;

    toGlClipSpaceDepth();
}
)";

static constexpr const char* const SRC3_WORLD_FS = R"(
in vec3 v_color;
in vec3 v_uv_z;
flat in vec4 v_lightDimMode;
flat in ivec2 v_texWinPos;
flat in ivec2 v_texWinSize;
flat in ivec2 v_clutPos;
flat in vec4 v_stmul;

float getLightDiminishingMultiplier(float z, vec4 lightDimMode) {
    float minLightIntensity = 64.0;
    float maxLightIntensity = lightDimMode.w;
    float offDimIntensity = 128.0;
    float floorDimIntensity = maxLightIntensity - z * 0.5;
    float wallDimintensity = ((128.0 * 65536.0) / z) / 256.0;

    float intensity = (
        offDimIntensity * lightDimMode.x +
        wallDimintensity * lightDimMode.y +
        floorDimIntensity * lightDimMode.z
    );

    intensity = clamp(intensity, minLightIntensity, maxLightIntensity);
    return intensity / 128.0;
}

void main() {
    out_color = tex8bpp(
        ivec2(floor(v_uv_z.xy)), v_texWinPos, v_texWinSize, v_clutPos, v_stmul, (WRAP_TEXTURE != 0), (ALLOW_DISCARD != 0)
    );

    vec3 colorMul = trunc(v_color * getLightDiminishingMultiplier(v_uv_z.z, v_lightDimMode) + 0.0001) / 128.0;
    colorMul = min(colorMul, 255.0 / 128.0);
    out_color.rgb *= colorMul;

#if USE_PSX_16_BIT_SHADING
    out_color = psxR5G5B5BitCrush(out_color);
#endif
}
)";

// sky.vert, sky.frag
static constexpr const char* const SRC3_SKY_VS = R"(
flat out vec2 v_uv_offset;
flat out ivec2 v_texWinPos;
flat out ivec2 v_texWinSize;
flat out ivec2 v_clutPos;

void main() {
    gl_Position = uniforms.mvpMatrix * vec4(in_pos, 1.0);
    gl_Position.y -= in_uv.y * max(gl_Position.w, 0.0) * 2.0;

    v_uv_offset = vec2(in_uv.x, 0.0);
    v_texWinPos = ivec2(in_texWinPos);
    v_texWinSize = ivec2(in_texWinSize);
    v_clutPos = ivec2(in_clutPos);

    toGlClipSpaceDepth();
}
)";

static constexpr const char* const SRC3_SKY_FS = R"(
flat in vec2 v_uv_offset;
flat in ivec2 v_texWinPos;
flat in ivec2 v_texWinSize;
flat in ivec2 v_clutPos;

vec2 getPixelPsxCoords() {
    vec2 fragCoord = floor(gl_FragCoord.xy) + 0.5;
    vec2 ndc = (fragCoord * uniforms.invDrawRes) * 2.0;
    return (ndc - uniforms.psxNdcOffset) * uniforms.ndcToPsxScale;
}

void main() {
    vec2 uv = getPixelPsxCoords() + v_uv_offset;

    if (uv.y >= float(v_texWinSize.y))
        discard;

    out_color = tex8bpp(ivec2(floor(uv)), v_texWinPos, v_texWinSize, v_clutPos, vec4(1.0, 1.0, 1.0, 1.0), true, false);

#if USE_PSX_16_BIT_SHADING
    out_color = psxR5G5B5BitCrush(out_color);
#endif
}
)";

// Post shaders: a quad drawn as a 4 vertex triangle strip, from its corners (0,0 = top left, 1,1 = bottom right).
// Note: the images drawn by the main render path have their first row at the top (the Vulkan layout).
static constexpr const char* const SRC3_QUAD_VS = R"(
layout(location = 0) in vec2 in_corner;

uniform vec4 u_posRect;     // Left x, top y, right x, bottom y (NDC)
uniform vec4 u_uvRect;      // Left u, top v, right u, bottom v

out vec2 v_uv;

void main() {
    gl_Position = vec4(mix(u_posRect.xy, u_posRect.zw, in_corner), 0.0, 1.0);
    v_uv = mix(u_uvRect.xy, u_uvRect.zw, in_corner);
}
)";

// Gamma adjustment through the 256 entry remap table, read the way the Vulkan renderer's nearest 1D texture sampling reads it
static constexpr const char* const SRC3_POST_FS_COMMON = R"(
uniform highp sampler2D gammaLut;

in vec2 v_uv;
layout(location = 0) out vec4 out_color;

vec3 gammaAdjust(vec3 color) {
    ivec3 idx = min(ivec3(floor(clamp(color, 0.0, 1.0) * 256.0)), ivec3(255));

    return vec3(
        texelFetch(gammaLut, ivec2(idx.r, 0), 0).r,
        texelFetch(gammaLut, ivec2(idx.g, 0), 0).r,
        texelFetch(gammaLut, ivec2(idx.b, 0), 0).r
    );
}
)";

static constexpr const char* const SRC3_PRESENT_FS = R"(
uniform highp sampler2D tex;

void main() {
    out_color = texture(tex, v_uv);

#if GAMMA_ADJUST
    out_color.rgb = gammaAdjust(out_color.rgb);
#endif
}
)";

// The PSX framebuffer as is (R16UI, PSX 5-5-5 pixels): decoded here instead of on the CPU
static constexpr const char* const SRC3_PSX_PRESENT_FS = R"(
uniform highp usampler2D psxFb;

void main() {
    ivec2 size = textureSize(psxFb, 0);
    ivec2 texel = clamp(ivec2(floor(v_uv * vec2(size))), ivec2(0, 0), size - 1);
    uint bits = texelFetch(psxFb, texel, 0).r;

    out_color = vec4(
        float(bits & 0x1Fu),
        float((bits >> 5u) & 0x1Fu),
        float((bits >> 10u) & 0x1Fu),
        31.0
    ) / 31.0;

#if GAMMA_ADJUST
    out_color.rgb = gammaAdjust(out_color.rgb);
#endif
}
)";

static constexpr const char* const SRC3_CROSSFADE_FS = R"(
uniform highp sampler2D tex1;
uniform highp sampler2D tex2;
uniform float u_lerp;

void main() {
    out_color = mix(texture(tex1, v_uv), texture(tex2, v_uv), u_lerp);

#if USE_PSX_16_BIT_SHADING
    out_color = trunc(out_color * 31.0 + 0.0001) / 31.0;
#endif

#if GAMMA_ADJUST
    out_color.rgb = gammaAdjust(out_color.rgb);
#endif
}
)";

static constexpr const char* const SRC3_QUAD_FS = R"(
uniform highp sampler2D tex;

void main() {
    out_color = texture(tex, v_uv);

#if MASKED
    if (out_color.a <= 0.0)
        discard;
#endif

#if GAMMA_ADJUST
    out_color.rgb = gammaAdjust(out_color.rgb);
#endif
}
)";

//==========================================================================================================================================
// OpenGL ES 2.0: GLSL ES 1.00
//==========================================================================================================================================
static constexpr const char* const SRC2_HEADER_VS = R"(#version 100
precision highp float;
)";

static constexpr const char* const SRC2_HEADER_FS_HIGHP = R"(#version 100
precision highp float;
#define SAMPLER_PRECISION highp
)";

static constexpr const char* const SRC2_HEADER_FS_MEDIUMP = R"(#version 100
precision mediump float;
#define SAMPLER_PRECISION mediump
)";

// Uniforms of the draw shaders: the fields of 'VShaderUniforms_Draw' (no uniform blocks)
static constexpr const char* const SRC2_DRAW_UNIFORMS = R"(
uniform mat4 u_mvpMatrix;
uniform vec2 u_ndcToPsxScale;
uniform vec2 u_psxNdcOffset;
uniform vec2 u_invDrawRes;
)";

// Vertex inputs of the draw shaders: 'VVertex_Draw', the integer attributes converted to floats
static constexpr const char* const SRC2_DRAW_VS_INPUTS = R"(
attribute vec3 in_pos;
attribute vec4 in_color_lightDimFlags;
attribute vec2 in_uv;
attribute vec2 in_texWinPos;
attribute vec2 in_texWinSize;
attribute vec2 in_clutPos;
attribute vec4 in_stmul;

// Vulkan clip space depth is 0 to w, OpenGL's is -w to w: remap so that geometry is clipped by the near and far planes as with Vulkan
void toGlClipSpaceDepth() {
    gl_Position.z = gl_Position.z * 2.0 - gl_Position.w;
}
)";

// Common fragment shader functions: reading texels from PSX VRAM. VRAM is 2 bytes per texel: low byte in red, high byte in
// 'VRAM_HI_CHANNEL'. Everything is done with floats holding small integers, which is exact.
static constexpr const char* const SRC2_DRAW_FS_COMMON = R"(
uniform SAMPLER_PRECISION sampler2D vramTex;    // A texture containing the entirety of PSX VRAM
uniform vec2 u_vramInvSize;                     // 1 / the size of PSX VRAM in texels

// The two bytes of a 16-bit texel: x = low byte, y = high byte
vec2 readVramBytes(vec2 xy) {
    vec4 texel = texture2D(vramTex, (xy + 0.5) * u_vramInvSize);
    return floor(vec2(texel.r, texel.VRAM_HI_CHANNEL) * 255.0 + 0.5);
}

vec4 sampleVramTexel(vec2 xy, vec4 stmul, bool bAllowDiscard) {
    vec2 texelBytes = readVramBytes(xy);

    if (bAllowDiscard && (texelBytes.x + texelBytes.y == 0.0))
        discard;

    vec4 color = vec4(
        mod(texelBytes.x, 32.0),                                            // Bits 0-4
        floor(texelBytes.x / 32.0) + mod(texelBytes.y, 4.0) * 8.0,          // Bits 5-9
        mod(floor(texelBytes.y / 4.0), 32.0),                               // Bits 10-14
        31.0
    ) / 31.0;

    if (texelBytes.y >= 128.0) {                                            // Bit 15: semi-transparent
        color *= stmul;
    }

    return color;
}

vec2 wrapOrClampUv(vec2 uv, vec2 texWinSize, bool bWrap, float wrapRange) {
    // Note: 'mod' is always positive, like the masking of the integer coordinates with Vulkan
    return (bWrap) ? mod(mod(uv, wrapRange), texWinSize) : clamp(uv, vec2(0.0, 0.0), texWinSize - 1.0);
}

vec4 tex16bpp(vec2 uv, vec2 texWinPos, vec2 texWinSize, vec4 stmul, bool bWrap, bool bAllowDiscard) {
    uv = wrapOrClampUv(uv, texWinSize, bWrap, 16777216.0) + texWinPos;
    return sampleVramTexel(uv, stmul, bAllowDiscard);
}

vec4 tex8bpp(vec2 uv, vec2 texWinPos, vec2 texWinSize, vec2 clutPos, vec4 stmul, bool bWrap, bool bAllowDiscard) {
    uv = wrapOrClampUv(uv, texWinSize, bWrap, 65536.0) + texWinPos;

    vec2 texelBytes = readVramBytes(vec2(floor(uv.x / 2.0), uv.y));
    float clutIdx = (mod(uv.x, 2.0) >= 1.0) ? texelBytes.y : texelBytes.x;
    return sampleVramTexel(vec2(clutPos.x + clutIdx, clutPos.y), stmul, bAllowDiscard);
}

vec4 tex4bpp(vec2 uv, vec2 texWinPos, vec2 texWinSize, vec2 clutPos, vec4 stmul, bool bWrap, bool bAllowDiscard) {
    uv = wrapOrClampUv(uv, texWinSize, bWrap, 65536.0) + texWinPos;

    vec2 texelBytes = readVramBytes(vec2(floor(uv.x / 4.0), uv.y));
    float nibbleIdx = mod(uv.x, 4.0);
    float texelByte = (nibbleIdx >= 2.0) ? texelBytes.y : texelBytes.x;
    float clutIdx = (mod(nibbleIdx, 2.0) >= 1.0) ? floor(texelByte / 16.0) : mod(texelByte, 16.0);
    return sampleVramTexel(vec2(clutPos.x + clutIdx, clutPos.y), stmul, bAllowDiscard);
}

vec4 psxR5G5B5BitCrush(vec4 color) {
    return floor(color * 31.0 + 0.0001) / 31.0;
}

// Values which are the same at every vertex, rounded back to that after interpolation
vec2 roundToInt(vec2 v) { return floor(v + 0.5); }
vec4 roundToInt(vec4 v) { return floor(v + 0.5); }
vec4 roundStmul(vec4 stmul) { return floor(stmul * 128.0 + 0.5) / 128.0; }
)";

static constexpr const char* const SRC2_COLORED_VS = R"(
varying vec4 v_color;

void main() {
    gl_Position = u_mvpMatrix * vec4(in_pos, 1.0);
    v_color = vec4(in_color_lightDimFlags.rgb, in_stmul.a) / 255.0;
}
)";

static constexpr const char* const SRC2_COLORED_FS = R"(
varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
)";

static constexpr const char* const SRC2_UI_VS = R"(
varying vec3 v_color;
varying vec2 v_uv;
varying vec4 v_texWin;      // Position, size
varying vec2 v_clutPos;
varying vec4 v_stmul;

void main() {
    gl_Position = u_mvpMatrix * vec4(in_pos, 1.0);
    v_color = in_color_lightDimFlags.rgb / 128.0;
    v_uv = in_uv;
    v_texWin = vec4(in_texWinPos, in_texWinSize);
    v_clutPos = in_clutPos;
    v_stmul = in_stmul / 128.0;
}
)";

static constexpr const char* const SRC2_UI_FS = R"(
varying vec3 v_color;
varying vec2 v_uv;
varying vec4 v_texWin;
varying vec2 v_clutPos;
varying vec4 v_stmul;

void main() {
    vec4 texWin = roundToInt(v_texWin);
    vec2 clutPos = roundToInt(v_clutPos);
    vec4 stmul = roundStmul(v_stmul);

#if TEX_BPP == 4
    vec4 color = tex4bpp(floor(v_uv), texWin.xy, texWin.zw, clutPos, stmul, false, true);
#elif TEX_BPP == 8
    vec4 color = tex8bpp(floor(v_uv), texWin.xy, texWin.zw, clutPos, stmul, false, true);
#else
    vec4 color = tex16bpp(floor(v_uv), texWin.xy, texWin.zw, stmul, false, true);
#endif

    color.rgb *= v_color;

#if USE_PSX_16_BIT_SHADING
    color = psxR5G5B5BitCrush(color);
#endif

    gl_FragColor = color;
}
)";

static constexpr const char* const SRC2_WORLD_VS = R"(
varying vec3 v_color;
varying vec3 v_uv_z;
varying vec4 v_lightDimMode;
varying vec4 v_texWin;      // Position, size
varying vec2 v_clutPos;
varying vec4 v_stmul;

bool hasFlag(float flags, float flag) {
    return (mod(floor(flags / flag), 2.0) >= 1.0);
}

void main() {
    gl_Position = u_mvpMatrix * vec4(in_pos, 1.0);

    float lightDimFlags = in_color_lightDimFlags.a;
    v_color = in_color_lightDimFlags.rgb;
    v_uv_z = vec3(in_uv, gl_Position.z);
    v_lightDimMode.x = (hasFlag(lightDimFlags, 1.0)) ? 1.0 : 0.0;       // No light diminishing
    v_lightDimMode.y = (hasFlag(lightDimFlags, 2.0)) ? 1.0 : 0.0;       // Wall light diminishing
    v_lightDimMode.z = (hasFlag(lightDimFlags, 4.0)) ? 1.0 : 0.0;       // Floor light diminishing
    v_lightDimMode.w = (hasFlag(lightDimFlags, 8.0)) ? 192.0 : 160.0;   // Max light diminishing intensity (stronger for Alpha 0.05)
    v_texWin = vec4(in_texWinPos, in_texWinSize);
    v_clutPos = in_clutPos;
    v_stmul = in_stmul / 128.0;

    toGlClipSpaceDepth();
}
)";

static constexpr const char* const SRC2_WORLD_FS = R"(
varying vec3 v_color;
varying vec3 v_uv_z;
varying vec4 v_lightDimMode;
varying vec4 v_texWin;
varying vec2 v_clutPos;
varying vec4 v_stmul;

float getLightDiminishingMultiplier(float z, vec4 lightDimMode) {
    float minLightIntensity = 64.0;
    float maxLightIntensity = lightDimMode.w;
    float offDimIntensity = 128.0;
    float floorDimIntensity = maxLightIntensity - z * 0.5;
    float wallDimintensity = ((128.0 * 65536.0) / z) / 256.0;

    float intensity = (
        offDimIntensity * lightDimMode.x +
        wallDimintensity * lightDimMode.y +
        floorDimIntensity * lightDimMode.z
    );

    intensity = clamp(intensity, minLightIntensity, maxLightIntensity);
    return intensity / 128.0;
}

void main() {
    vec4 texWin = roundToInt(v_texWin);
    vec4 color = tex8bpp(
        floor(v_uv_z.xy), texWin.xy, texWin.zw, roundToInt(v_clutPos), roundStmul(v_stmul), (WRAP_TEXTURE != 0), (ALLOW_DISCARD != 0)
    );

    vec3 colorMul = floor(v_color * getLightDiminishingMultiplier(v_uv_z.z, roundToInt(v_lightDimMode)) + 0.0001) / 128.0;
    colorMul = min(colorMul, 255.0 / 128.0);
    color.rgb *= colorMul;

#if USE_PSX_16_BIT_SHADING
    color = psxR5G5B5BitCrush(color);
#endif

    gl_FragColor = color;
}
)";

static constexpr const char* const SRC2_SKY_VS = R"(
varying float v_uOffset;
varying vec4 v_texWin;      // Position, size
varying vec2 v_clutPos;

void main() {
    gl_Position = u_mvpMatrix * vec4(in_pos, 1.0);
    gl_Position.y -= in_uv.y * max(gl_Position.w, 0.0) * 2.0;

    v_uOffset = in_uv.x;
    v_texWin = vec4(in_texWinPos, in_texWinSize);
    v_clutPos = in_clutPos;

    toGlClipSpaceDepth();
}
)";

static constexpr const char* const SRC2_SKY_FS = R"(
varying float v_uOffset;
varying vec4 v_texWin;
varying vec2 v_clutPos;

vec2 getPixelPsxCoords() {
    vec2 fragCoord = floor(gl_FragCoord.xy) + 0.5;
    vec2 ndc = (fragCoord * u_invDrawRes) * 2.0;
    return (ndc - u_psxNdcOffset) * u_ndcToPsxScale;
}

void main() {
    vec4 texWin = roundToInt(v_texWin);
    vec2 uv = getPixelPsxCoords() + vec2(v_uOffset, 0.0);

    if (uv.y >= texWin.w)
        discard;

    vec4 color = tex8bpp(floor(uv), texWin.xy, texWin.zw, roundToInt(v_clutPos), vec4(1.0, 1.0, 1.0, 1.0), true, false);

#if USE_PSX_16_BIT_SHADING
    color = psxR5G5B5BitCrush(color);
#endif

    gl_FragColor = color;
}
)";

// Post shaders: as for OpenGL ES 3.0
static constexpr const char* const SRC2_QUAD_VS = R"(
attribute vec2 in_corner;

uniform vec4 u_posRect;     // Left x, top y, right x, bottom y (NDC)
uniform vec4 u_uvRect;      // Left u, top v, right u, bottom v

varying vec2 v_uv;

void main() {
    gl_Position = vec4(mix(u_posRect.xy, u_posRect.zw, in_corner), 0.0, 1.0);
    v_uv = mix(u_uvRect.xy, u_uvRect.zw, in_corner);
}
)";

static constexpr const char* const SRC2_POST_FS_COMMON = R"(
uniform SAMPLER_PRECISION sampler2D gammaLut;

varying vec2 v_uv;

float gammaRemap(float value) {
    float idx = min(floor(clamp(value, 0.0, 1.0) * 256.0), 255.0);
    return texture2D(gammaLut, vec2((idx + 0.5) / 256.0, 0.5)).r;
}

vec3 gammaAdjust(vec3 color) {
    return vec3(gammaRemap(color.r), gammaRemap(color.g), gammaRemap(color.b));
}
)";

static constexpr const char* const SRC2_PRESENT_FS = R"(
uniform SAMPLER_PRECISION sampler2D tex;

void main() {
    vec4 color = texture2D(tex, v_uv);

#if GAMMA_ADJUST
    color.rgb = gammaAdjust(color.rgb);
#endif

    gl_FragColor = color;
}
)";

// The PSX framebuffer as is (2 bytes per pixel, PSX 5-5-5 pixels): decoded here instead of on the CPU
static constexpr const char* const SRC2_PSX_PRESENT_FS = R"(
uniform SAMPLER_PRECISION sampler2D psxFb;

void main() {
    vec4 texel = texture2D(psxFb, v_uv);
    vec2 texelBytes = floor(vec2(texel.r, texel.VRAM_HI_CHANNEL) * 255.0 + 0.5);

    vec4 color = vec4(
        mod(texelBytes.x, 32.0),
        floor(texelBytes.x / 32.0) + mod(texelBytes.y, 4.0) * 8.0,
        mod(floor(texelBytes.y / 4.0), 32.0),
        31.0
    ) / 31.0;

#if GAMMA_ADJUST
    color.rgb = gammaAdjust(color.rgb);
#endif

    gl_FragColor = color;
}
)";

static constexpr const char* const SRC2_CROSSFADE_FS = R"(
uniform SAMPLER_PRECISION sampler2D tex1;
uniform SAMPLER_PRECISION sampler2D tex2;
uniform float u_lerp;

void main() {
    vec4 color = mix(texture2D(tex1, v_uv), texture2D(tex2, v_uv), u_lerp);

#if USE_PSX_16_BIT_SHADING
    color = floor(color * 31.0 + 0.0001) / 31.0;
#endif

#if GAMMA_ADJUST
    color.rgb = gammaAdjust(color.rgb);
#endif

    gl_FragColor = color;
}
)";

static constexpr const char* const SRC2_QUAD_FS = R"(
uniform SAMPLER_PRECISION sampler2D tex;

void main() {
    vec4 color = texture2D(tex, v_uv);

#if MASKED
    if (color.a <= 0.0)
        discard;
#endif

#if GAMMA_ADJUST
    color.rgb = gammaAdjust(color.rgb);
#endif

    gl_FragColor = color;
}
)";

//==========================================================================================================================================
// Program definitions
//==========================================================================================================================================
enum class ProgramKind : uint8_t {
    Draw,   // Draw programs: 'VVertex_Draw' inputs, the draw uniforms and PSX VRAM
    Post    // Post programs: quad corners as inputs, their own uniforms
};

struct ProgramDef {
    ProgramKind     kind;
    const char*     defines;
    const char*     vs3;
    const char*     fs3;
    const char*     vs2;
    const char*     fs2;
};

static constexpr ProgramDef PROGRAM_DEFS[(uint32_t) GProgram::NUM_PROGRAMS] = {
    { ProgramKind::Draw, "",                                                    SRC3_COLORED_VS,    SRC3_COLORED_FS,        SRC2_COLORED_VS,    SRC2_COLORED_FS     },  // Colored
    { ProgramKind::Draw, "#define TEX_BPP 4\n",                                 SRC3_UI_VS,         SRC3_UI_FS,             SRC2_UI_VS,         SRC2_UI_FS          },  // UI_4bpp
    { ProgramKind::Draw, "#define TEX_BPP 8\n",                                 SRC3_UI_VS,         SRC3_UI_FS,             SRC2_UI_VS,         SRC2_UI_FS          },  // UI_8bpp
    { ProgramKind::Draw, "#define TEX_BPP 16\n",                                SRC3_UI_VS,         SRC3_UI_FS,             SRC2_UI_VS,         SRC2_UI_FS          },  // UI_16bpp
    { ProgramKind::Draw, "#define WRAP_TEXTURE 1\n#define ALLOW_DISCARD 1\n",   SRC3_WORLD_VS,      SRC3_WORLD_FS,          SRC2_WORLD_VS,      SRC2_WORLD_FS       },  // World_Wrap
    { ProgramKind::Draw, "#define WRAP_TEXTURE 1\n#define ALLOW_DISCARD 0\n",   SRC3_WORLD_VS,      SRC3_WORLD_FS,          SRC2_WORLD_VS,      SRC2_WORLD_FS       },  // World_WrapOpaque
    { ProgramKind::Draw, "#define WRAP_TEXTURE 0\n#define ALLOW_DISCARD 1\n",   SRC3_WORLD_VS,      SRC3_WORLD_FS,          SRC2_WORLD_VS,      SRC2_WORLD_FS       },  // World_Clamp
    { ProgramKind::Draw, "",                                                    SRC3_SKY_VS,        SRC3_SKY_FS,            SRC2_SKY_VS,        SRC2_SKY_FS         },  // Sky
    { ProgramKind::Post, "#define GAMMA_ADJUST 0\n",                            SRC3_QUAD_VS,       SRC3_PRESENT_FS,        SRC2_QUAD_VS,       SRC2_PRESENT_FS     },  // Present
    { ProgramKind::Post, "#define GAMMA_ADJUST 1\n",                            SRC3_QUAD_VS,       SRC3_PRESENT_FS,        SRC2_QUAD_VS,       SRC2_PRESENT_FS     },  // Present_Gamma
    { ProgramKind::Post, "#define GAMMA_ADJUST 0\n",                            SRC3_QUAD_VS,       SRC3_PSX_PRESENT_FS,    SRC2_QUAD_VS,       SRC2_PSX_PRESENT_FS },  // PsxPresent
    { ProgramKind::Post, "#define GAMMA_ADJUST 1\n",                            SRC3_QUAD_VS,       SRC3_PSX_PRESENT_FS,    SRC2_QUAD_VS,       SRC2_PSX_PRESENT_FS },  // PsxPresent_Gamma
    { ProgramKind::Post, "#define GAMMA_ADJUST 0\n",                            SRC3_QUAD_VS,       SRC3_CROSSFADE_FS,      SRC2_QUAD_VS,       SRC2_CROSSFADE_FS   },  // Crossfade
    { ProgramKind::Post, "#define GAMMA_ADJUST 1\n",                            SRC3_QUAD_VS,       SRC3_CROSSFADE_FS,      SRC2_QUAD_VS,       SRC2_CROSSFADE_FS   },  // Crossfade_Gamma
    { ProgramKind::Post, "#define GAMMA_ADJUST 0\n#define MASKED 0\n",          SRC3_QUAD_VS,       SRC3_QUAD_FS,           SRC2_QUAD_VS,       SRC2_QUAD_FS        },  // Quad
    { ProgramKind::Post, "#define GAMMA_ADJUST 1\n#define MASKED 0\n",          SRC3_QUAD_VS,       SRC3_QUAD_FS,           SRC2_QUAD_VS,       SRC2_QUAD_FS        },  // Quad_Gamma
    { ProgramKind::Post, "#define GAMMA_ADJUST 0\n#define MASKED 1\n",          SRC3_QUAD_VS,       SRC3_QUAD_FS,           SRC2_QUAD_VS,       SRC2_QUAD_FS        },  // QuadMasked
    { ProgramKind::Post, "#define GAMMA_ADJUST 1\n#define MASKED 1\n",          SRC3_QUAD_VS,       SRC3_QUAD_FS,           SRC2_QUAD_VS,       SRC2_QUAD_FS        },  // QuadMasked_Gamma
};

// Vertex attribute names and their locations (OpenGL ES 2.0 binds them before linking, the GLSL ES 3.00 shaders declare them)
static constexpr struct { GLuint location; const char* name; } ATTRIB_BINDINGS[] = {
    { ATTRIB_POS,               "in_pos"                    },
    { ATTRIB_COLOR_LIGHTDIM,    "in_color_lightDimFlags"    },
    { ATTRIB_UV,                "in_uv"                     },
    { ATTRIB_TEXWIN_POS,        "in_texWinPos"              },
    { ATTRIB_TEXWIN_SIZE,       "in_texWinSize"             },
    { ATTRIB_CLUT_POS,          "in_clutPos"                },
    { ATTRIB_STMUL,             "in_stmul"                  },
    { ATTRIB_CORNER,            "in_corner"                 },
};

static GProgramInfo     gPrograms[(uint32_t) GProgram::NUM_PROGRAMS];
static GApi             gApi;

//------------------------------------------------------------------------------------------------------------------------------------------
// Builds the full source of a shader of a program
//------------------------------------------------------------------------------------------------------------------------------------------
static std::string makeShaderSource(const ProgramDef& def, const bool bVertexShader, const GShaderOptions& options) noexcept {
    const bool bGles3 = (options.api == GApi::Gles3);
    std::string src;
    src.reserve(8 * 1024);

    if (bGles3) {
        src += SRC3_HEADER;
    } else if (bVertexShader) {
        src += SRC2_HEADER_VS;
    } else {
        src += (options.bHighpFragment) ? SRC2_HEADER_FS_HIGHP : SRC2_HEADER_FS_MEDIUMP;
    }

    src += def.defines;
    src += (options.bUse16BitShading) ? "#define USE_PSX_16_BIT_SHADING 1\n" : "#define USE_PSX_16_BIT_SHADING 0\n";

    if (!bGles3) {
        src += (options.vramHiChannel == 'a') ? "#define VRAM_HI_CHANNEL a\n" : "#define VRAM_HI_CHANNEL g\n";
    }

    const char* const fsBody = (bGles3) ? def.fs3 : def.fs2;
    const bool bColoredFs = ((fsBody == SRC3_COLORED_FS) || (fsBody == SRC2_COLORED_FS));

    if (def.kind == ProgramKind::Draw) {
        src += (bGles3) ? SRC3_DRAW_UNIFORMS : SRC2_DRAW_UNIFORMS;

        if (bVertexShader) {
            src += (bGles3) ? SRC3_DRAW_VS_INPUTS : SRC2_DRAW_VS_INPUTS;
        } else if (!bColoredFs) {
            src += (bGles3) ? SRC3_DRAW_FS_COMMON : SRC2_DRAW_FS_COMMON;
        }
    } else if (!bVertexShader) {
        src += (bGles3) ? SRC3_POST_FS_COMMON : SRC2_POST_FS_COMMON;
    }

    if (bVertexShader) {
        src += (bGles3) ? def.vs3 : def.vs2;
    } else {
        src += fsBody;
    }

    return src;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Program binaries: core in OpenGL ES 3.0, 'OES_get_program_binary' in OpenGL ES 2.0
//------------------------------------------------------------------------------------------------------------------------------------------
static void setProgramBinary(const GLuint program, const GLenum format, const void* const pData, const GLsizei size) noexcept {
    if (gApi == GApi::Gles3) {
        glProgramBinary(program, format, pData, size);
    } else {
        glProgramBinaryOES(program, format, pData, size);
    }
}

static void getProgramBinary(const GLuint program, const GLsizei bufSize, GLsizei& length, GLenum& format, void* const pData) noexcept {
    if (gApi == GApi::Gles3) {
        glGetProgramBinary(program, bufSize, &length, &format, pData);
    } else {
        glGetProgramBinaryOES(program, bufSize, &length, &format, pData);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Gets the info log of a shader or program
//------------------------------------------------------------------------------------------------------------------------------------------
static std::string getShaderInfoLog(const GLuint shader) noexcept {
    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::string log((size_t) std::max(logLength, 1), '\0');
    glGetShaderInfoLog(shader, (GLsizei) log.size(), nullptr, log.data());
    return log;
}

static std::string getProgramInfoLog(const GLuint program) noexcept {
    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::string log((size_t) std::max(logLength, 1), '\0');
    glGetProgramInfoLog(program, (GLsizei) log.size(), nullptr, log.data());
    return log;
}

static bool isProgramLinked(const GLuint program) noexcept {
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    return (linkStatus == GL_TRUE);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the uniform block binding and texture units of a linked program and gets its uniform locations.
// Needed after every link and every binary loaded: both reset that state.
//------------------------------------------------------------------------------------------------------------------------------------------
static void setupProgram(GProgramInfo& info, const ProgramDef& def) noexcept {
    const GLuint program = info.program;
    glUseProgram(program);

    const auto setTexUnit = [=](const char* const name, const GLuint unit) noexcept {
        const GLint location = glGetUniformLocation(program, name);

        if (location >= 0) {
            glUniform1i(location, (GLint) unit);
        }
    };

    if (def.kind == ProgramKind::Draw) {
        if (gApi == GApi::Gles3) {
            const GLuint blockIdx = glGetUniformBlockIndex(program, "DrawUniforms");

            if (blockIdx != GL_INVALID_INDEX) {
                glUniformBlockBinding(program, blockIdx, DRAW_UNIFORMS_BINDING);
            }
        }

        setTexUnit("vramTex", TEX_UNIT_VRAM);
    } else {
        setTexUnit("tex", TEX_UNIT_SRC);
        setTexUnit("tex1", TEX_UNIT_SRC);
        setTexUnit("tex2", TEX_UNIT_SRC2);
        setTexUnit("psxFb", TEX_UNIT_SRC);
        setTexUnit("gammaLut", TEX_UNIT_GAMMA_LUT);
    }

    info.uPosRect = glGetUniformLocation(program, "u_posRect");
    info.uUvRect = glGetUniformLocation(program, "u_uvRect");
    info.uLerp = glGetUniformLocation(program, "u_lerp");
    info.uMvpMatrix = glGetUniformLocation(program, "u_mvpMatrix");
    info.uNdcToPsxScale = glGetUniformLocation(program, "u_ndcToPsxScale");
    info.uPsxNdcOffset = glGetUniformLocation(program, "u_psxNdcOffset");
    info.uInvDrawRes = glGetUniformLocation(program, "u_invDrawRes");
    info.uVramInvSize = glGetUniformLocation(program, "u_vramInvSize");
    glUseProgram(0);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates all of the programs for the given API path.
// Programs are taken from the cache where possible; the rest are all compiled and linked before any status is checked, so that drivers
// can compile them in parallel (KHR_parallel_shader_compile, or their own background compiler threads).
//------------------------------------------------------------------------------------------------------------------------------------------
void init(const GShaderOptions& options) noexcept {
    constexpr uint32_t NUM_PROGRAMS = (uint32_t) GProgram::NUM_PROGRAMS;
    gApi = options.api;

    struct PendingProgram {
        std::string     vsSrc;
        std::string     fsSrc;
        std::string     cacheKey;
        GLuint          vs;
        GLuint          fs;
        bool            bNeedsCompile;
    };

    std::vector<PendingProgram> pending(NUM_PROGRAMS);

    if (GLAD_GL_KHR_parallel_shader_compile && glMaxShaderCompilerThreadsKHR) {
        glMaxShaderCompilerThreadsKHR(0xFFFFFFFFu);     // As many threads as the driver wants
    }

    // Sources and cache keys, then try the cache
    uint32_t numPrograms = 0;
    uint32_t numFromCache = 0;

    for (uint32_t i = 0; i < NUM_PROGRAMS; ++i) {
        const ProgramDef& def = PROGRAM_DEFS[i];
        PendingProgram& pend = pending[i];
        GProgramInfo& info = gPrograms[i];
        info = {};

        if ((def.kind == ProgramKind::Draw) && (!options.bMakeDrawPrograms))
            continue;

        ++numPrograms;
        pend.vsSrc = makeShaderSource(def, true, options);
        pend.fsSrc = makeShaderSource(def, false, options);
        pend.bNeedsCompile = true;
        info.program = glCreateProgram();

        if (!options.bUseProgramBinaries)
            continue;

        const char* const sources[2] = { pend.vsSrc.c_str(), pend.fsSrc.c_str() };
        pend.cacheKey = GShaderCache::makeKey(sources, 2);

        GShaderCache::ProgramBinary binary = {};

        if (!GShaderCache::find(pend.cacheKey, binary))
            continue;

        setProgramBinary(info.program, binary.format, binary.pData->data(), (GLsizei) binary.pData->size());

        if (isProgramLinked(info.program)) {
            pend.bNeedsCompile = false;
            ++numFromCache;
        } else {
            // Rejected by the driver (it changed in a way the key does not cover): compile it again and replace the binary
            GShaderCache::remove(pend.cacheKey);
            glDeleteProgram(info.program);
            info.program = glCreateProgram();
        }
    }

    // Compile and link everything not cached, without waiting on any of it
    for (uint32_t i = 0; i < NUM_PROGRAMS; ++i) {
        PendingProgram& pend = pending[i];
        const GLuint program = gPrograms[i].program;

        if ((!program) || (!pend.bNeedsCompile))
            continue;

        const char* const pVsSrc = pend.vsSrc.c_str();
        const char* const pFsSrc = pend.fsSrc.c_str();

        pend.vs = glCreateShader(GL_VERTEX_SHADER);
        pend.fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(pend.vs, 1, &pVsSrc, nullptr);
        glShaderSource(pend.fs, 1, &pFsSrc, nullptr);
        glCompileShader(pend.vs);
        glCompileShader(pend.fs);
        glAttachShader(program, pend.vs);
        glAttachShader(program, pend.fs);

        for (const auto& binding : ATTRIB_BINDINGS) {
            glBindAttribLocation(program, binding.location, binding.name);
        }

        if (options.bUseProgramBinaries && (gApi == GApi::Gles3)) {
            glProgramParameteri(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
        }

        glLinkProgram(program);
    }

    // Check the results and cache the binaries of the new programs
    for (uint32_t i = 0; i < NUM_PROGRAMS; ++i) {
        PendingProgram& pend = pending[i];
        GProgramInfo& info = gPrograms[i];

        if (!info.program)
            continue;

        if (pend.bNeedsCompile) {
            if (!isProgramLinked(info.program)) {
                const std::string vsLog = getShaderInfoLog(pend.vs);
                const std::string fsLog = getShaderInfoLog(pend.fs);
                const std::string programLog = getProgramInfoLog(info.program);
                SDL_Log("PsyDoom GLES: shader program %u failed:\nVS:\n%s\nFS:\n%s\nProgram:\n%s", i, vsLog.c_str(), fsLog.c_str(), programLog.c_str());
                FatalErrors::raiseF("GLES renderer: failed to build shader program %u!\n%s\n%s\n%s", i, vsLog.c_str(), fsLog.c_str(), programLog.c_str());
            }

            glDetachShader(info.program, pend.vs);
            glDetachShader(info.program, pend.fs);
            glDeleteShader(pend.vs);
            glDeleteShader(pend.fs);

            if (options.bUseProgramBinaries) {
                GLint binaryLength = 0;
                glGetProgramiv(info.program, GL_PROGRAM_BINARY_LENGTH, &binaryLength);  // Note: same enum as 'GL_PROGRAM_BINARY_LENGTH_OES'

                if (binaryLength > 0) {
                    auto pData = std::make_shared<std::vector<uint8_t>>((size_t) binaryLength);
                    GLenum binaryFormat = 0;
                    GLsizei lengthOut = 0;
                    getProgramBinary(info.program, binaryLength, lengthOut, binaryFormat, pData->data());

                    if (lengthOut > 0) {
                        pData->resize((size_t) lengthOut);
                        GShaderCache::add(pend.cacheKey, GShaderCache::ProgramBinary{ binaryFormat, std::move(pData) });
                    }
                }
            }
        }

        setupProgram(info, PROGRAM_DEFS[i]);
    }

    SDL_Log(
        "PsyDoom GLES: %u shader programs (%s), %u from the shader cache",
        numPrograms,
        (gApi == GApi::Gles3) ? "GLSL ES 3.00" : "GLSL ES 1.00",
        numFromCache
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Destroys all programs. If the GL context was lost they are gone already, so only their names are forgotten.
//------------------------------------------------------------------------------------------------------------------------------------------
void destroy(const bool bContextLost) noexcept {
    for (GProgramInfo& info : gPrograms) {
        if (info.program && (!bContextLost)) {
            glDeleteProgram(info.program);
        }

        info = {};
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// OpenGL ES 2.0: gives the draw programs the size of PSX VRAM (texture coordinates are normalized there)
//------------------------------------------------------------------------------------------------------------------------------------------
void setVramSize(const uint32_t vramW, const uint32_t vramH) noexcept {
    for (uint32_t i = 0; i < NUM_DRAW_PROGRAMS; ++i) {
        const GProgramInfo& info = gPrograms[i];

        if (info.program && (info.uVramInvSize >= 0)) {
            glUseProgram(info.program);
            glUniform2f(info.uVramInvSize, 1.0f / (float) vramW, 1.0f / (float) vramH);
        }
    }

    glUseProgram(0);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Gets a program and its info
//------------------------------------------------------------------------------------------------------------------------------------------
const GProgramInfo& get(const GProgram program) noexcept {
    ASSERT((uint32_t) program < (uint32_t) GProgram::NUM_PROGRAMS);
    return gPrograms[(uint32_t) program];
}

END_NAMESPACE(GShaders)

#endif  // #if PSYDOOM_GLES_RENDERER
