#version 460

//----------------------------------------------------------------------------------------------------------------------
// World shader: fragment.
// This shader handles all in-game walls, floors, and sprites with or without light diminishing effects.
// All textures are also assumed to be 8 bits per pixel.
//----------------------------------------------------------------------------------------------------------------------
#include "ShaderCommon_Frag.h"

layout(constant_id = 0) const bool WRAP_TEXTURE = true;             // Whether to use wrap texture mode or clamp to edge; clamping is used for sprites to prevent edge artifacts
layout(constant_id = 1) const bool USE_PSX_16_BIT_SHADING = true;   // Whether to shade with 16-bit precision like the original PlayStation

// Whether a '0' texel discards the fragment (PSX masking). Pipelines for geometry whose texture uses no CLUT entry that is '0'
// turn this off: the result is identical, and without 'discard' in the shader tile based GPUs can remove the hidden fragments of
// everything drawn underneath (Mali forward pixel kill and early ZS, Adreno LRZ, PowerVR HSR) instead of shading all of them.
layout(constant_id = 2) const bool ALLOW_DISCARD = true;

// A texture containing the entirety of PSX VRAM
layout(set = 0, binding = 0) uniform usampler2D vramTex;

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_uv_z;
layout(location = 2) flat in vec4 in_lightDimMode;
layout(location = 3) flat in ivec2 in_texWinPos;
layout(location = 4) flat in ivec2 in_texWinSize;
layout(location = 5) flat in ivec2 in_clutPos;
layout(location = 6) flat in vec4 in_stmul;

layout(location = 0) out vec4 out_color;

//----------------------------------------------------------------------------------------------------------------------
// Compute the light diminishing multiplier for a pixel Z depth and strength vector for the different diminish styles.
// We use a strength vector here to avoid 'if()' branching when supporting the different light diminishing modes.
//----------------------------------------------------------------------------------------------------------------------
float getLightDiminishingMultiplier(float z, vec4 in_lightDimMode) {
    // Minimum and maximum light diminishing intensity
    float minLightIntensity = 64;
    float maxLightIntensity = in_lightDimMode.w;

    // This is the light diminishing intensity when the effect is off (no change)
    float offDimIntensity = 128.0;

    // Compute the light diminishing intensity for floors
    float floorDimIntensity = maxLightIntensity - z * 0.5;

    // Compute the light diminishing intensity for walls
    float wallDimintensity = ((128 * 65536) / z) / 256;

    // Compute the light diminishing intensity we will use
    float intensity = (
        offDimIntensity * in_lightDimMode.x +
        wallDimintensity * in_lightDimMode.y + 
        floorDimIntensity * in_lightDimMode.z
    );

    // Clamp the intensity to the min/max allowed amounts (0.5x to 1.25x in normalized coords).
    // Then scale the diminish intensity back to normalized color coords rather than 0-128.
    //
    // Note: prior to normalization here I originally truncated the intensity to an integer to try and mimic the classic
    // renderer's calculations more closely. I'm finding now however this now causes blocky artifacts when dual colored
    // lighting is mixed with this light diminishing effect - the mixing of the two gradients is not smooth and block
    // artifacts appear. Leaving the intensity in unquantized format instead makes the banding smoother, curved, and less
    // noticeable. It also doesn't appear to cause any (visible) extra difference between the classic and Vulkan renderers.
    //
    intensity = clamp(intensity, minLightIntensity, maxLightIntensity);
    return intensity / 128.0;
}

//----------------------------------------------------------------------------------------------------------------------
// Shader entrypoint: do shading for a world/3d-view texel, with or without light diminishing
//----------------------------------------------------------------------------------------------------------------------
void main() {
    // Sample the raw texel first
    out_color = tex8bpp(vramTex, ivec2(floor(in_uv_z.xy)), in_texWinPos, in_texWinSize, in_clutPos, in_stmul, WRAP_TEXTURE, ALLOW_DISCARD);

    // Compute color multiply after accounting for input color and light diminishing effects.
    // Add a little bias also to prevent switching back and forth between cases that are close, due to float inprecision...
    vec3 colorMul = trunc(in_color * getLightDiminishingMultiplier(in_uv_z.z, in_lightDimMode) + 0.0001) / 128.0;

    // The PSX renderer doesn't allow the color multiply to go larger than this:
    colorMul = min(colorMul, 255.0 / 128.0);

    // Apply the color multiply and bit crush the color in a manner similar to the PSX
    out_color.rgb *= colorMul;

    if (USE_PSX_16_BIT_SHADING) {
        out_color = psxR5G5B5BitCrush(out_color);
    }
}
