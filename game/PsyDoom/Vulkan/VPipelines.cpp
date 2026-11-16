//------------------------------------------------------------------------------------------------------------------------------------------
// Logic for creating the various shaders and pipelines used by the new Vulkan renderer
//------------------------------------------------------------------------------------------------------------------------------------------
#include "VPipelines.h"

#if PSYDOOM_VULKAN_RENDERER

#include "DescriptorSetLayout.h"
#include "FatalErrors.h"
#include "LogicalDevice.h"
#include "PhysicalDevice.h"
#include "Pipeline.h"
#include "PipelineLayout.h"
#include "PsyDoom/Config/Config.h"
#include "PsyDoom/PlayerPrefs.h"
#include "Sampler.h"
#include "ShaderModule.h"
#include "VRenderer.h"
#include "VRenderPath_Crossfade.h"
#include "VRenderPath_LoadingPlaque.h"
#include "VRenderPath_Main.h"
#include "VRenderPath_Psx.h"

BEGIN_NAMESPACE(VPipelines)

// The raw SPIRV binary code for the shaders
#include "SPIRV_colored_frag.bin.h"
#include "SPIRV_colored_vert.bin.h"
#include "SPIRV_crossfade_frag.bin.h"
#include "SPIRV_crossfade_gamma_frag.bin.h"
#include "SPIRV_gamma_adjust_blit_frag.bin.h"
#include "SPIRV_gamma_adjust_post_process_frag.bin.h"
#include "SPIRV_msaa_resolve_frag.bin.h"
#include "SPIRV_ndc_position_only_vert.bin.h"
#include "SPIRV_ndc_textured_frag.bin.h"
#include "SPIRV_ndc_textured_vert.bin.h"
#include "SPIRV_sky_frag.bin.h"
#include "SPIRV_sky_vert.bin.h"
#include "SPIRV_ui_16bpp_frag.bin.h"
#include "SPIRV_ui_4bpp_frag.bin.h"
#include "SPIRV_ui_8bpp_frag.bin.h"
#include "SPIRV_ui_vert.bin.h"
#include "SPIRV_world_frag.bin.h"
#include "SPIRV_world_vert.bin.h"

// Vertex binding descriptions
const VkVertexInputBindingDescription gVertexBindingDesc_draw           = { 0, sizeof(VVertex_Draw), VK_VERTEX_INPUT_RATE_VERTEX };
const VkVertexInputBindingDescription gVertexBindingDesc_msaaResolve    = { 0, sizeof(VVertex_MsaaResolve), VK_VERTEX_INPUT_RATE_VERTEX };
const VkVertexInputBindingDescription gVertexBindingDesc_xyUv           = { 0, sizeof(VVertex_XyUv), VK_VERTEX_INPUT_RATE_VERTEX };

// Vertex attribute descriptions
const VkVertexInputAttributeDescription gVertexAttribs_draw[] = {
    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VVertex_Draw, x) },
    { 1, 0, VK_FORMAT_R8G8B8A8_UINT,    offsetof(VVertex_Draw, r) },
    { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(VVertex_Draw, u) },
    { 3, 0, VK_FORMAT_R16G16_UINT,      offsetof(VVertex_Draw, texWinX) },
    { 4, 0, VK_FORMAT_R16G16_UINT,      offsetof(VVertex_Draw, texWinW) },
    { 5, 0, VK_FORMAT_R16G16_UINT,      offsetof(VVertex_Draw, clutX) },
    { 6, 0, VK_FORMAT_R8G8B8A8_UINT,    offsetof(VVertex_Draw, stmulR) },
};

const VkVertexInputAttributeDescription gVertexAttribs_msaaResolve[] = {
    { 0, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(VVertex_MsaaResolve, x) },
};

const VkVertexInputAttributeDescription gVertexAttribs_xyUv[] = {
    { 0, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(VVertex_XyUv, x) },
    { 1, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(VVertex_XyUv, u) },
};

// Shaders: see the associated source files for more comments/details
static vgl::ShaderModule    gShader_colored_vert;
static vgl::ShaderModule    gShader_colored_frag;
static vgl::ShaderModule    gShader_ui_vert;
static vgl::ShaderModule    gShader_ui_16bpp_frag;
static vgl::ShaderModule    gShader_ui_4bpp_frag;
static vgl::ShaderModule    gShader_ui_8bpp_frag;
static vgl::ShaderModule    gShader_world_vert;
static vgl::ShaderModule    gShader_world_frag;
static vgl::ShaderModule    gShader_sky_vert;
static vgl::ShaderModule    gShader_sky_frag;
static vgl::ShaderModule    gShader_ndc_position_only_vert;
static vgl::ShaderModule    gShader_ndc_textured_vert;
static vgl::ShaderModule    gShader_ndc_textured_frag;
static vgl::ShaderModule    gShader_crossfade_frag;
static vgl::ShaderModule    gShader_crossfade_gamma_frag;
static vgl::ShaderModule    gShader_gamma_adjust_blit_frag;
static vgl::ShaderModule    gShader_gamma_adjust_post_process_frag;
static vgl::ShaderModule    gShader_msaa_resolve_frag;

// Sets of shader modules
vgl::ShaderModule* const gShaders_colored[]                 = { &gShader_colored_vert, &gShader_colored_frag };
vgl::ShaderModule* const gShaders_ui_4bpp[]                 = { &gShader_ui_vert, &gShader_ui_4bpp_frag };
vgl::ShaderModule* const gShaders_ui_8bpp[]                 = { &gShader_ui_vert, &gShader_ui_8bpp_frag };
vgl::ShaderModule* const gShaders_ui_16bpp[]                = { &gShader_ui_vert, &gShader_ui_16bpp_frag };
vgl::ShaderModule* const gShaders_world[]                   = { &gShader_world_vert, &gShader_world_frag };
vgl::ShaderModule* const gShaders_sky[]                     = { &gShader_sky_vert, &gShader_sky_frag };
vgl::ShaderModule* const gShaders_ndcTextured[]             = { &gShader_ndc_textured_vert, &gShader_ndc_textured_frag };
vgl::ShaderModule* const gShaders_crossfade[]               = { &gShader_ndc_textured_vert, &gShader_crossfade_frag };
vgl::ShaderModule* const gShaders_crossfade_gamma[]         = { &gShader_ndc_textured_vert, &gShader_crossfade_gamma_frag };
vgl::ShaderModule* const gShaders_gammaAdjustBlit[]         = { &gShader_ndc_textured_vert, &gShader_gamma_adjust_blit_frag };
vgl::ShaderModule* const gShaders_gammaAdjustPostProcess[]  = { &gShader_ndc_position_only_vert, &gShader_gamma_adjust_post_process_frag };
vgl::ShaderModule* const gShaders_msaaResolve[]             = { &gShader_ndc_position_only_vert, &gShader_msaa_resolve_frag };

// Pipeline samplers
vgl::Sampler gSampler_draw;
vgl::Sampler gSampler_normClampNearest;

// Pipeline descriptor set layouts
vgl::DescriptorSetLayout gDescSetLayout_draw;               // Used by all the normal drawing pipelines
vgl::DescriptorSetLayout gDescSetLayout_draw_noTex;         // For drawing without a texture
vgl::DescriptorSetLayout gDescSetLayout_blit1Tex;           // Used to blit an image using 1 texture
vgl::DescriptorSetLayout gDescSetLayout_blit2Tex;           // Used to blit an image using 2 textures
vgl::DescriptorSetLayout gDescSetLayout_blit3Tex;           // Used to blit an image using 3 textures
vgl::DescriptorSetLayout gDescSetLayout_postProcess0Tex;    // Post process an input attachment with 0 additional input textures
vgl::DescriptorSetLayout gDescSetLayout_postProcess1Tex;    // Post process an input attachment with 1 additional input texture

// Pipeline layouts
vgl::PipelineLayout gPipelineLayout_draw;               // Used by all the normal drawing pipelines
vgl::PipelineLayout gPipelineLayout_draw_noTex;         // For drawing without a texture
vgl::PipelineLayout gPipelineLayout_blit1Tex;           // Used to blit an image using 1 texture
vgl::PipelineLayout gPipelineLayout_blit2Tex;           // Used to blit an image using 2 textures
vgl::PipelineLayout gPipelineLayout_postProcess0Tex;    // Post process an input attachment with 0 additional input textures
vgl::PipelineLayout gPipelineLayout_postProcess1Tex;    // Post process an input attachment with 1 additional input texture
vgl::PipelineLayout gPipelineLayout_crossfade;          // For drawing crossfades
vgl::PipelineLayout gPipelineLayout_crossfadeGamma;     // For drawing crossfades with gamma adjustment

// Pipeline input assembly states
vgl::PipelineInputAssemblyState gInputAS_lineList;      // A list of lines
vgl::PipelineInputAssemblyState gInputAS_triList;       // A list of triangles

// Pipeline rasterization states
vgl::PipelineRasterizationState gRasterState_noCull;
vgl::PipelineRasterizationState gRasterState_backFaceCull;

// Pipeline multisample states
vgl::PipelineMultisampleState gMultisampleState_noMultisample;
vgl::PipelineMultisampleState gMultisampleState_perSettings;
vgl::PipelineMultisampleState gMultisampleState_perSettingsEdgeOnly;    // Same as 'per settings' but don't multi-sample interior triangle texels

// Pipeline color blend per-attachment states
VkPipelineColorBlendAttachmentState gBlendAS_noBlend;           // No blending
VkPipelineColorBlendAttachmentState gBlendAS_alpha;             // Regular alpha blending
VkPipelineColorBlendAttachmentState gBlendAS_additive;          // Additive blending
VkPipelineColorBlendAttachmentState gBlendAS_subtractive;       // Subtractive blending

// Pipeline color blend state for all attachments: note that all these assume blending ONLY on attachment at index '0'!
vgl::PipelineColorBlendState gBlendState_noBlend;
vgl::PipelineColorBlendState gBlendState_alpha;
vgl::PipelineColorBlendState gBlendState_additive;
vgl::PipelineColorBlendState gBlendState_subtractive;

// Pipeline depth/stencil states
vgl::PipelineDepthStencilState gDepthState_disabled;        // No depth/stencil buffer: Depth write, test and all stencil operations disabled

// The pipelines themselves
VPipelineSet<VPipelineType_Main>            gPipelines_Main_NoGammaAdjust;
VPipelineSet<VPipelineType_Main>            gPipelines_Main_GammaAdjust;
VPipelineSet<VPipelineType_Main>            gPipelines_Main_SwapchainResolve;
VPipelineSet<VPipelineType_Crossfade>       gPipelines_Crossfade;
VPipelineSet<VPipelineType_LoadingPlaque>   gPipelines_LoadingPlaque;
VPipelineSet<VPipelineType_PSX>             gPipelines_PSX;

//------------------------------------------------------------------------------------------------------------------------------------------
// Helper: destroys all pipelines in the specified pipeline set
//------------------------------------------------------------------------------------------------------------------------------------------
template <class PipelineSetT>
static void destroyPipelineSet(PipelineSetT& pipelineSet) noexcept {
    for (vgl::Pipeline& pipeline : pipelineSet.pipelines) {
        pipeline.destroy(true);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initialize a single shader and raise a fatal error if it fails
//-----------------------------------------------------------------------------------------------------------------------------------------
static void initShader(
    vgl::LogicalDevice& device,
    vgl::ShaderModule& shader,
    const VkShaderStageFlagBits stageFlags,
    const uint32_t* const pCode,
    const uint32_t codeSize,
    const char* const shaderName
) noexcept {
    if (!shader.init(device, stageFlags, pCode, codeSize))
        FatalErrors::raiseF("Failed to init Vulkan shader '%s'!", shaderName);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all shaders
//------------------------------------------------------------------------------------------------------------------------------------------
static void initShaders(vgl::LogicalDevice& device) noexcept {
    initShader(device, gShader_colored_vert, VK_SHADER_STAGE_VERTEX_BIT, gSPIRV_colored_vert, sizeof(gSPIRV_colored_vert), "colored_vert");
    initShader(device, gShader_colored_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_colored_frag, sizeof(gSPIRV_colored_frag), "colored_frag");
    initShader(device, gShader_ui_vert, VK_SHADER_STAGE_VERTEX_BIT, gSPIRV_ui_vert, sizeof(gSPIRV_ui_vert), "ui_vert");
    initShader(device, gShader_ui_4bpp_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_ui_4bpp_frag, sizeof(gSPIRV_ui_4bpp_frag), "ui_4bpp_frag");
    initShader(device, gShader_ui_8bpp_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_ui_8bpp_frag, sizeof(gSPIRV_ui_8bpp_frag), "ui_8bpp_frag");
    initShader(device, gShader_ui_16bpp_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_ui_16bpp_frag, sizeof(gSPIRV_ui_16bpp_frag), "ui_16bpp_frag");
    initShader(device, gShader_world_vert, VK_SHADER_STAGE_VERTEX_BIT, gSPIRV_world_vert, sizeof(gSPIRV_world_vert), "world_vert");
    initShader(device, gShader_world_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_world_frag, sizeof(gSPIRV_world_frag), "world_frag");
    initShader(device, gShader_sky_vert, VK_SHADER_STAGE_VERTEX_BIT, gSPIRV_sky_vert, sizeof(gSPIRV_sky_vert), "sky_vert");
    initShader(device, gShader_sky_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_sky_frag, sizeof(gSPIRV_sky_frag), "sky_frag");
    initShader(device, gShader_ndc_position_only_vert, VK_SHADER_STAGE_VERTEX_BIT, gSPIRV_ndc_position_only_vert, sizeof(gSPIRV_ndc_position_only_vert), "ndc_position_only_vert");
    initShader(device, gShader_ndc_textured_vert, VK_SHADER_STAGE_VERTEX_BIT, gSPIRV_ndc_textured_vert, sizeof(gSPIRV_ndc_textured_vert), "ndc_textured_vert");
    initShader(device, gShader_ndc_textured_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_ndc_textured_frag, sizeof(gSPIRV_ndc_textured_frag), "ndc_textured_frag");
    initShader(device, gShader_crossfade_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_crossfade_frag, sizeof(gSPIRV_crossfade_frag), "crossfade_frag");
    initShader(device, gShader_crossfade_gamma_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_crossfade_gamma_frag, sizeof(gSPIRV_crossfade_gamma_frag), "crossfade_gamma_frag");
    initShader(device, gShader_gamma_adjust_blit_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_gamma_adjust_blit_frag, sizeof(gSPIRV_gamma_adjust_blit_frag), "gamma_adjust_blit_frag");
    initShader(device, gShader_gamma_adjust_post_process_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_gamma_adjust_post_process_frag, sizeof(gSPIRV_gamma_adjust_post_process_frag), "gamma_adjust_post_process_frag");
    initShader(device, gShader_msaa_resolve_frag, VK_SHADER_STAGE_FRAGMENT_BIT, gSPIRV_msaa_resolve_frag, sizeof(gSPIRV_msaa_resolve_frag), "msaa_resolve_frag");
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all samplers
//------------------------------------------------------------------------------------------------------------------------------------------
static void initSamplers(vgl::LogicalDevice& device) noexcept {
    // Sampler used for most drawing operations
    {
        vgl::SamplerSettings settings = vgl::SamplerSettings().setToDefault();
        settings.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        settings.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        settings.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        settings.minLod = 0;
        settings.maxLod = 0;
        settings.unnormalizedCoordinates = true;    // Note: Vulkan requires min/max lod to be '0' and clamp to edge to be 'true' if using un-normalized coords

        if (!gSampler_draw.init(device, settings))
            FatalErrors::raise("Failed to init a Vulkan sampler!");
    }

    // A sampler with nearest neighbor filtering, clamp to edge and using normalized coordinates
    {
        vgl::SamplerSettings settings = vgl::SamplerSettings().setToDefault();
        settings.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        settings.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        settings.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        settings.minLod = 0;
        settings.maxLod = 0;

        if (!gSampler_normClampNearest.init(device, settings))
            FatalErrors::raise("Failed to init a Vulkan sampler!");
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all descriptor set layouts.
// Requires all samplers to have been created first and uses immutable samplers baked into the layout for better performance.
//------------------------------------------------------------------------------------------------------------------------------------------
static void initDescriptorSetLayouts(vgl::LogicalDevice& device) noexcept {
    // Regular drawing
    {
         const VkSampler vkSamplers[] = { gSampler_draw.getVkSampler() };

        VkDescriptorSetLayoutBinding bindings[1] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = C_ARRAY_SIZE(vkSamplers);
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[0].pImmutableSamplers = vkSamplers;

        if (!gDescSetLayout_draw.init(device, bindings, C_ARRAY_SIZE(bindings)))
            FatalErrors::raise("Failed to init the 'Draw' Vulkan descriptor set layout!");
    }
    
    // Drawing without a texture
    if (!gDescSetLayout_draw_noTex.init(device, nullptr, 0))
        FatalErrors::raise("Failed to init the 'Draw' (No texture) Vulkan descriptor set layout!");
    
    // Blit (1 texture)
    {
        const VkSampler vkSampler = gSampler_normClampNearest.getVkSampler();

        VkDescriptorSetLayoutBinding bindings[1] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[0].pImmutableSamplers = &vkSampler;

        if (!gDescSetLayout_blit1Tex.init(device, bindings, C_ARRAY_SIZE(bindings)))
            FatalErrors::raise("Failed to init the 'Blit (1 texture)' Vulkan descriptor set layout!");
    }
    
    // Blit (2 texture)
    {
        const VkSampler vkSampler = gSampler_normClampNearest.getVkSampler();

        // Note: used to use an array of 2 textures, but MoltenVK didn't like that on MacOS.
        // Use 2 separate texture bindings instead to work around the issue...
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[0].pImmutableSamplers = &vkSampler;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].pImmutableSamplers = &vkSampler;

        if (!gDescSetLayout_blit2Tex.init(device, bindings, C_ARRAY_SIZE(bindings)))
            FatalErrors::raise("Failed to init the 'Blit (2 texture)' Vulkan descriptor set layout!");
    }
    
    // Blit (3 texture)
    {
        const VkSampler vkSampler = gSampler_normClampNearest.getVkSampler();

        // Note: used to use an array of 2 textures, but MoltenVK didn't like that on MacOS.
        // Use 3 separate texture bindings instead to work around the issue...
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[0].pImmutableSamplers = &vkSampler;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].pImmutableSamplers = &vkSampler;
        
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[2].pImmutableSamplers = &vkSampler;

        if (!gDescSetLayout_blit3Tex.init(device, bindings, C_ARRAY_SIZE(bindings)))
            FatalErrors::raise("Failed to init the 'Blit (3 texture)' Vulkan descriptor set layout!");
    }

    // Post process (0 texture)
    {
        VkDescriptorSetLayoutBinding bindings[1] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        if (!gDescSetLayout_postProcess0Tex.init(device, bindings, C_ARRAY_SIZE(bindings)))
            FatalErrors::raise("Failed to init the 'Post process (0 texture)' Vulkan descriptor set layout!");
    }
    
    // Post process (1 texture)
    {
        const VkSampler vkSampler = gSampler_normClampNearest.getVkSampler();

        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].pImmutableSamplers = &vkSampler;

        if (!gDescSetLayout_postProcess1Tex.init(device, bindings, C_ARRAY_SIZE(bindings)))
            FatalErrors::raise("Failed to init the 'Post process (1 texture)' Vulkan descriptor set layout!");
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline layouts.
// Requires descriptor set layouts to have been created first.
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineLayouts(vgl::LogicalDevice& device) noexcept {
    // Regular drawing pipeline layout: uses push constants to set the MVP matrix.
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_draw.getVkLayout() };

        VkPushConstantRange uniformPushConstants = {};
        uniformPushConstants.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        uniformPushConstants.offset = 0;
        uniformPushConstants.size = sizeof(VShaderUniforms_Draw);

        if (!gPipelineLayout_draw.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), &uniformPushConstants, 1))
            FatalErrors::raise("Failed to init the 'Draw' Vulkan pipeline layout!");
    }
    
    // Drawing without a texture pipeline layout: uses push constants to set the MVP matrix.
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_draw_noTex.getVkLayout() };

        VkPushConstantRange uniformPushConstants = {};
        uniformPushConstants.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        uniformPushConstants.offset = 0;
        uniformPushConstants.size = sizeof(VShaderUniforms_Draw);

        if (!gPipelineLayout_draw_noTex.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), &uniformPushConstants, 1))
            FatalErrors::raise("Failed to init the 'Draw' (No texture) Vulkan pipeline layout!");
    }
    
    // Standard 'blit' and 'post process' pipeline layouts without any push constants
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_blit1Tex.getVkLayout() };

        if (!gPipelineLayout_blit1Tex.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), nullptr, 0))
            FatalErrors::raise("Failed to init the 'Blit (1 texture)' Vulkan pipeline layout!");
    }
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_blit2Tex.getVkLayout() };

        if (!gPipelineLayout_blit2Tex.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), nullptr, 0))
            FatalErrors::raise("Failed to init the 'Blit (2 texture)' Vulkan pipeline layout!");
    }
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_postProcess0Tex.getVkLayout() };

        if (!gPipelineLayout_postProcess0Tex.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), nullptr, 0))
            FatalErrors::raise("Failed to init the 'Post process (0 texture)' Vulkan pipeline layout!");
    }
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_postProcess1Tex.getVkLayout() };

        if (!gPipelineLayout_postProcess1Tex.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), nullptr, 0))
            FatalErrors::raise("Failed to init the 'Post process (1 texture)' Vulkan pipeline layout!");
    }
    
    // Crossfade pipeline layouts: uses push constants to set the lerp factor
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_blit2Tex.getVkLayout() };

        VkPushConstantRange uniformPushConstants = {};
        uniformPushConstants.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        uniformPushConstants.offset = 0;
        uniformPushConstants.size = sizeof(VShaderUniforms_Crossfade);

        if (!gPipelineLayout_crossfade.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), &uniformPushConstants, 1))
            FatalErrors::raise("Failed to init the 'Crossfade' Vulkan pipeline layout!");
    }
    {
        const VkDescriptorSetLayout vkDescSetLayouts[] = { gDescSetLayout_blit3Tex.getVkLayout() };

        VkPushConstantRange uniformPushConstants = {};
        uniformPushConstants.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        uniformPushConstants.offset = 0;
        uniformPushConstants.size = sizeof(VShaderUniforms_Crossfade);

        if (!gPipelineLayout_crossfadeGamma.init(device, vkDescSetLayouts, C_ARRAY_SIZE(vkDescSetLayouts), &uniformPushConstants, 1))
            FatalErrors::raise("Failed to init the 'Crossfade (With Gamma Adjustment)' Vulkan pipeline layout!");
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline input assembly states
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineInputAssemblyStates() noexcept {
    gInputAS_lineList = vgl::PipelineInputAssemblyState().setToDefault();
    gInputAS_lineList.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    gInputAS_triList = vgl::PipelineInputAssemblyState().setToDefault();
    gInputAS_triList.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline rasterization states
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineRasterizationStates() noexcept {
    gRasterState_noCull = vgl::PipelineRasterizationState().setToDefault();

    gRasterState_backFaceCull = vgl::PipelineRasterizationState().setToDefault();
    gRasterState_backFaceCull.cullMode = VK_CULL_MODE_BACK_BIT;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline mulitsample states
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineMultisampleStates(vgl::LogicalDevice& device, const uint32_t numSamples) noexcept {
    // This state always has multisampling disabled
    gMultisampleState_noMultisample = vgl::PipelineMultisampleState().setToDefault();

    // This state *may* have a varying amount of multisampling depending on current graphic settings
    gMultisampleState_perSettings = vgl::PipelineMultisampleState().setToDefault();
    gMultisampleState_perSettings.rasterizationSamples = (VkSampleCountFlagBits) numSamples;

    // Android too, the same image as the other builds: the fragment shader runs once per sample (with 4x MSAA four times the
    // fragment work of edge only MSAA), the price of the textures being antialiased as well and not only the edges.
    if (device.getPhysicalDevice()->getFeatures().sampleRateShading) {
        // Enable sample rate shading if we can get it for nicer MSAA.
        // Force it to do MSAA for every single fragment to help eliminate texture shimmer and shader aliasing.
        gMultisampleState_perSettings.sampleShadingEnable = true;
        gMultisampleState_perSettings.minSampleShading = 1.0f;
    }

    // This is a variant of the 'per settings' state but without sample rate shading (edge only MSAA).
    // This mode is more appropriate for UI elements since MSAA can blur them slightly.
    gMultisampleState_perSettingsEdgeOnly = vgl::PipelineMultisampleState().setToDefault();
    gMultisampleState_perSettingsEdgeOnly.rasterizationSamples = (VkSampleCountFlagBits) numSamples;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline color blend attachment states
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineColorBlendAttachmentStates() noexcept {
    // No blending
    gBlendAS_noBlend = {};
    gBlendAS_noBlend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    gBlendAS_noBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    gBlendAS_noBlend.colorWriteMask = (
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT    // Don't need alpha writes but it's probably more efficient to write 32-bits at a time
    );

    // Common state for when we are blending
    VkPipelineColorBlendAttachmentState blendAS_blendCommon = {};
    blendAS_blendCommon.blendEnable = VK_TRUE;
    blendAS_blendCommon.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAS_blendCommon.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAS_blendCommon.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAS_blendCommon.colorWriteMask = (
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT    // Don't need alpha writes but it's probably more efficient to write 32-bits at a time
    );

    // Blending modes
    gBlendAS_alpha = blendAS_blendCommon;
    gBlendAS_alpha.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    gBlendAS_alpha.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    gBlendAS_alpha.colorBlendOp = VK_BLEND_OP_ADD;

    gBlendAS_additive = blendAS_blendCommon;
    gBlendAS_additive.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    gBlendAS_additive.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    gBlendAS_additive.colorBlendOp = VK_BLEND_OP_ADD;

    gBlendAS_subtractive = blendAS_blendCommon;
    gBlendAS_subtractive.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    gBlendAS_subtractive.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    gBlendAS_subtractive.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline color blend states
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineColorBlendStates() noexcept {
    // Note that all these assume blending ONLY on attachment at index '0'!
    gBlendState_noBlend = vgl::PipelineColorBlendState().setToDefault();
    gBlendState_noBlend.pAttachments = &gBlendAS_noBlend;
    gBlendState_noBlend.attachmentCount = 1;

    gBlendState_alpha = vgl::PipelineColorBlendState().setToDefault();
    gBlendState_alpha.pAttachments = &gBlendAS_alpha;
    gBlendState_alpha.attachmentCount = 1;

    gBlendState_additive = vgl::PipelineColorBlendState().setToDefault();
    gBlendState_additive.pAttachments = &gBlendAS_additive;
    gBlendState_additive.attachmentCount = 1;

    gBlendState_subtractive = vgl::PipelineColorBlendState().setToDefault();
    gBlendState_subtractive.pAttachments = &gBlendAS_subtractive;
    gBlendState_subtractive.attachmentCount = 1;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipeline depth stencil states
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineDepthStencilStates() noexcept {
    gDepthState_disabled = vgl::PipelineDepthStencilState().setToDefault();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes a single pipeline and raise a fatal error on failure
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipeline(
    vgl::Pipeline& pipeline,
    const vgl::RenderPass& renderPass,
    const uint32_t subpassIdx,
    const vgl::ShaderModule* const pShaderModules[2],
    const VkSpecializationInfo* const pShaderSpecializationInfo,
    const vgl::PipelineLayout& pipelineLayout,
    const VkVertexInputBindingDescription& vertexBindingDesc,
    const VkVertexInputAttributeDescription* const pVertexAttribs,
    const uint32_t numVertexAttribs,
    const vgl::PipelineInputAssemblyState& inputAssemblyState,
    const vgl::PipelineRasterizationState& rasterizerState,
    const vgl::PipelineColorBlendState& colorBlendState,
    const vgl::PipelineDepthStencilState& depthStencilState,
    const vgl::PipelineMultisampleState& multisampleState
) noexcept {
    const bool bSuccess = pipeline.initGraphicsPipeline(
        pipelineLayout,
        renderPass,
        subpassIdx,
        pShaderModules,
        2,                          // For this project always using just a vertex and fragment shader
        pShaderSpecializationInfo,
        &vertexBindingDesc,
        1,                          // For this project always a single input stream of vertices
        pVertexAttribs,
        numVertexAttribs,
        inputAssemblyState,
        rasterizerState,
        multisampleState,
        colorBlendState,
        depthStencilState
    );

    if (!bSuccess)
        FatalErrors::raise("Failed to create a Vulkan graphics pipeline used for rendering!");
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Convenience function: initialize a pipeline used for normal drawing and fill in a few parameters that are common to all draw pipelines
//------------------------------------------------------------------------------------------------------------------------------------------
static void initDrawPipeline(
    vgl::Pipeline& pipeline,
    const vgl::RenderPass& renderPass,
    const vgl::ShaderModule* const pShaderModules[2],
    const vgl::PipelineInputAssemblyState& inputAssemblyState,
    const vgl::PipelineRasterizationState& rasterizerState,
    const vgl::PipelineColorBlendState& colorBlendState,
    const vgl::PipelineDepthStencilState& depthStencilState,
    const bool bNoTexture,
    const bool bWrapTexture,
    const bool bEdgeOnlyMsaa,
    const bool bAllowDiscard = true
) noexcept {
    // Shader specialization constants.
    // Note: an entry for a constant a shader does not declare is ignored (only the world shader has 'ALLOW_DISCARD').
    struct ShaderSpecConsts {
        VkBool32 bWrapTexture;          // Whether to do wrapping (true) or clamping (false) when texturing
        VkBool32 bUse16BitShading;      // Whether to use the original PSX 16-bit shading or not
        VkBool32 bAllowDiscard;         // Whether a '0' texel discards the fragment (world shader)
    } shaderSpecConsts;

    shaderSpecConsts.bWrapTexture = bWrapTexture;
    shaderSpecConsts.bUse16BitShading = (!Config::gbUseVulkan32BitShading);
    shaderSpecConsts.bAllowDiscard = bAllowDiscard;

    const VkSpecializationMapEntry specializationMapEntries[] = {
        { 0, offsetof(ShaderSpecConsts, bWrapTexture), sizeof(shaderSpecConsts.bWrapTexture) },
        { 1, offsetof(ShaderSpecConsts, bUse16BitShading), sizeof(shaderSpecConsts.bUse16BitShading) },
        { 2, offsetof(ShaderSpecConsts, bAllowDiscard), sizeof(shaderSpecConsts.bAllowDiscard) },
    };

    VkSpecializationInfo specializationInfo = {};
    specializationInfo.mapEntryCount = C_ARRAY_SIZE(specializationMapEntries);
    specializationInfo.pMapEntries = specializationMapEntries;
    specializationInfo.dataSize = sizeof(ShaderSpecConsts);
    specializationInfo.pData = &shaderSpecConsts;

    // Create the pipeline
    initPipeline(
        pipeline,
        renderPass,
        0,
        pShaderModules,
        &specializationInfo,
        (bNoTexture) ? gPipelineLayout_draw_noTex : gPipelineLayout_draw,
        gVertexBindingDesc_draw,
        gVertexAttribs_draw,
        C_ARRAY_SIZE(gVertexAttribs_draw),
        inputAssemblyState,
        rasterizerState,
        colorBlendState,
        depthStencilState,
        (bEdgeOnlyMsaa) ? gMultisampleState_perSettingsEdgeOnly : gMultisampleState_perSettings
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes a pipeline set for the main render path
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineSet_Main(
    VPipelineSet<VPipelineType_Main>& pipelineSet,
    const vgl::RenderPass& renderPass,
    const uint32_t numSamples,
    const bool bGammaAdjusted,
    const bool bShaderMsaaResolve
) noexcept {
    // Create all of the main drawing pipelines
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::Lines), renderPass,
        gShaders_colored, gInputAS_lineList, gRasterState_noCull, gBlendState_noBlend, gDepthState_disabled, true, false, true
    );
        
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::Colored), renderPass,
        gShaders_colored, gInputAS_triList, gRasterState_noCull, gBlendState_noBlend, gDepthState_disabled, true, false, true
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::UI_4bpp), renderPass,
        gShaders_ui_4bpp, gInputAS_triList, gRasterState_noCull, gBlendState_noBlend, gDepthState_disabled, false, false, true
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::UI_8bpp), renderPass,
        gShaders_ui_8bpp, gInputAS_triList, gRasterState_noCull, gBlendState_noBlend, gDepthState_disabled, false, false, true
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::UI_8bpp_Add), renderPass,
        gShaders_ui_8bpp, gInputAS_triList, gRasterState_noCull, gBlendState_additive, gDepthState_disabled, false, false, true
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::UI_16bpp), renderPass,
        gShaders_ui_16bpp, gInputAS_triList, gRasterState_noCull, gBlendState_noBlend, gDepthState_disabled, false, false, true
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_GeomMasked), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_backFaceCull, gBlendState_noBlend, gDepthState_disabled, false, true, false
    );

    // The same for walls and flats whose texture uses no '0' CLUT entry: no discard in the shader (see 'RV_OpaqueGeomPipelineFor')
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_GeomOpaque), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_backFaceCull, gBlendState_noBlend, gDepthState_disabled, false, true, false, false
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_GeomAlpha), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_backFaceCull, gBlendState_alpha, gDepthState_disabled, false, true, false
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_SpriteMasked), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_noCull, gBlendState_noBlend, gDepthState_disabled, false, false, false
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_SpriteAlpha), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_noCull, gBlendState_alpha, gDepthState_disabled, false, false, false
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_SpriteAdditive), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_noCull, gBlendState_additive, gDepthState_disabled, false, false, false
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_SpriteSubtractive), renderPass,
        gShaders_world, gInputAS_triList, gRasterState_noCull, gBlendState_subtractive, gDepthState_disabled, false, false, false
    );
    
    initDrawPipeline(pipelineSet.get(VPipelineType_Main::World_Sky), renderPass,
        gShaders_sky, gInputAS_triList, gRasterState_backFaceCull, gBlendState_noBlend, gDepthState_disabled, false, true, true
    );

    // The pipeline to resolve MSAA: only bother creating this if we are doing MSAA resolved by a shader (not by the hardware).
    // Specialize the shader to the number of samples also, so that loops can be unrolled.
    ASSERT((!bShaderMsaaResolve) || (numSamples > 1));

    if (bShaderMsaaResolve) {
        const VkSpecializationMapEntry specializationMapEntries[] = {
            { 0, 0, sizeof(uint32_t) }
        };

        VkSpecializationInfo specializationInfo = {};
        specializationInfo.mapEntryCount = C_ARRAY_SIZE(specializationMapEntries);
        specializationInfo.pMapEntries = specializationMapEntries;
        specializationInfo.dataSize = sizeof(uint32_t);
        specializationInfo.pData = &numSamples;

        initPipeline(
            pipelineSet.get(VPipelineType_Main::MsaaResolve), renderPass, 1, // Subpass index = 1
            gShaders_msaaResolve, &specializationInfo, gPipelineLayout_postProcess0Tex,
            gVertexBindingDesc_msaaResolve, gVertexAttribs_msaaResolve, C_ARRAY_SIZE(gVertexAttribs_msaaResolve),
            gInputAS_triList, gRasterState_noCull,
            gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
        );
    }
    
    // The pipeline used to adjust gamma for an input attachment.
    // Only define this for the gamma adjusted render pass.
    if (bGammaAdjusted) {
        initPipeline(
            pipelineSet.get(VPipelineType_Main::GammaAdjustPostProcess), renderPass, (bShaderMsaaResolve) ? 2 : 1, // Subpass index = 2 or 1 (after the resolve subpass, if any)
            gShaders_gammaAdjustPostProcess, nullptr, gPipelineLayout_postProcess1Tex,
            gVertexBindingDesc_xyUv, gVertexAttribs_xyUv, C_ARRAY_SIZE(gVertexAttribs_xyUv),
            gInputAS_triList, gRasterState_noCull,
            gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
        );
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes a pipeline set for the crossfade render path
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineSet_Crossfade(VPipelineSet<VPipelineType_Crossfade>& pipelineSet, const vgl::RenderPass& renderPass) noexcept {
    // Using 32-bit shading for the Vulkan renderer?
    const VkBool32 bShade16Bit = (!Config::gbUseVulkan32BitShading);
    
    // Specializations for the crossfade pipelines (whether to use 16 or 32-bit shading)
    const VkSpecializationMapEntry vkSpecMapEntries_crossfade[] = {
        { 0, 0, sizeof(VkBool32) }
    };
    
    VkSpecializationInfo vkSpecInfo_crossfade = {};
    vkSpecInfo_crossfade.mapEntryCount = C_ARRAY_SIZE(vkSpecMapEntries_crossfade);
    vkSpecInfo_crossfade.pMapEntries = vkSpecMapEntries_crossfade;
    vkSpecInfo_crossfade.dataSize = sizeof(VkBool32);
    vkSpecInfo_crossfade.pData = &bShade16Bit;

    // Crossfade pipelines
    initPipeline(
        pipelineSet.get(VPipelineType_Crossfade::Crossfade), renderPass, 0,
        gShaders_crossfade, &vkSpecInfo_crossfade, gPipelineLayout_crossfade,
        gVertexBindingDesc_xyUv, gVertexAttribs_xyUv, C_ARRAY_SIZE(gVertexAttribs_xyUv),
        gInputAS_triList, gRasterState_noCull,
        gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
    );
    
    initPipeline(
        pipelineSet.get(VPipelineType_Crossfade::CrossfadeGammaAdjusted), renderPass, 0,
        gShaders_crossfade_gamma, &vkSpecInfo_crossfade, gPipelineLayout_crossfadeGamma,
        gVertexBindingDesc_xyUv, gVertexAttribs_xyUv, C_ARRAY_SIZE(gVertexAttribs_xyUv),
        gInputAS_triList, gRasterState_noCull,
        gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes a pipeline set for the loading plaque render path.
// These are used to draw loading plaques, including the background behind them.
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineSet_LoadingPlaque(VPipelineSet<VPipelineType_LoadingPlaque>& pipelineSet, const vgl::RenderPass& renderPass) noexcept {
    initPipeline(
        pipelineSet.get(VPipelineType_LoadingPlaque::LoadingPlaque), renderPass, 0,
        gShaders_ndcTextured, nullptr, gPipelineLayout_blit1Tex,
        gVertexBindingDesc_xyUv, gVertexAttribs_xyUv, C_ARRAY_SIZE(gVertexAttribs_xyUv),
        gInputAS_triList, gRasterState_noCull,
        gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
    );
    
    initPipeline(
        pipelineSet.get(VPipelineType_LoadingPlaque::LoadingPlaqueGammaAdjusted), renderPass, 0,
        gShaders_gammaAdjustBlit, nullptr, gPipelineLayout_blit2Tex,
        gVertexBindingDesc_xyUv, gVertexAttribs_xyUv, C_ARRAY_SIZE(gVertexAttribs_xyUv),
        gInputAS_triList, gRasterState_noCull,
        gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes a pipeline set for the PSX render path
//------------------------------------------------------------------------------------------------------------------------------------------
static void initPipelineSet_PSX(VPipelineSet<VPipelineType_PSX>& pipelineSet, const vgl::RenderPass& renderPass) noexcept {
    // Used for doing a gamma adjusted blit
    initPipeline(
        pipelineSet.get(VPipelineType_PSX::GammaAdjustBlit), renderPass, 0,
        gShaders_gammaAdjustBlit, nullptr, gPipelineLayout_blit2Tex,
        gVertexBindingDesc_xyUv, gVertexAttribs_xyUv, C_ARRAY_SIZE(gVertexAttribs_xyUv),
        gInputAS_triList, gRasterState_noCull,
        gBlendState_noBlend, gDepthState_disabled, gMultisampleState_noMultisample
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all of the building blocks that make up pipelines, but not the pipelines themselves.
// These elements have very few dependencies and can be initialized early in order to solve bootstrap dependency issues.
//------------------------------------------------------------------------------------------------------------------------------------------
void initPipelineComponents(vgl::LogicalDevice& device, const uint32_t numSamples) noexcept {
    // Create all pipeline creation inputs and states
    initShaders(device);
    initSamplers(device);
    initDescriptorSetLayouts(device);
    initPipelineLayouts(device);
    initPipelineInputAssemblyStates();
    initPipelineRasterizationStates();
    initPipelineMultisampleStates(device, numSamples);
    initPipelineColorBlendAttachmentStates();
    initPipelineColorBlendStates();
    initPipelineDepthStencilStates();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes all pipelines.
// This step must be done after all pipeline components have been created.
//------------------------------------------------------------------------------------------------------------------------------------------
void initPipelines(
    VRenderPath_Main& mainRPath,
    VRenderPath_Psx& psxRPath,
    VRenderPath_Crossfade& crossfadeRPath,
    VRenderPath_LoadingPlaque& loadingPlaqueRPath,
    const uint32_t numSamples
) noexcept
{
    const bool bShaderMsaaResolve = ((numSamples > 1) && (!mainRPath.usesHardwareMsaaResolve()));
    initPipelineSet_Main(gPipelines_Main_NoGammaAdjust, mainRPath.getRenderPass_NoGammaAdjust(), numSamples, false, bShaderMsaaResolve);
    initPipelineSet_Main(gPipelines_Main_GammaAdjust, mainRPath.getRenderPass_GammaAdjust(), numSamples, true, bShaderMsaaResolve);

    // MSAA resolved by the hardware straight into the swapchain image: another render pass (its second subpass draws nothing)
    if (mainRPath.canResolveToSwapchain()) {
        initPipelineSet_Main(gPipelines_Main_SwapchainResolve, mainRPath.getRenderPass_SwapchainResolve(), numSamples, false, false);
    }

    initPipelineSet_Crossfade(gPipelines_Crossfade, crossfadeRPath.getRenderPass());
    initPipelineSet_LoadingPlaque(gPipelines_LoadingPlaque, loadingPlaqueRPath.getRenderPass());
    initPipelineSet_PSX(gPipelines_PSX, psxRPath.getRenderPass());
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Destroys all pipelines and their components
//------------------------------------------------------------------------------------------------------------------------------------------
void shutdown() noexcept {
    destroyPipelineSet(gPipelines_PSX);
    destroyPipelineSet(gPipelines_LoadingPlaque);
    destroyPipelineSet(gPipelines_Crossfade);
    destroyPipelineSet(gPipelines_Main_SwapchainResolve);
    destroyPipelineSet(gPipelines_Main_GammaAdjust);
    destroyPipelineSet(gPipelines_Main_NoGammaAdjust);

    gPipelineLayout_crossfadeGamma.destroy(true);
    gPipelineLayout_crossfade.destroy(true);
    gPipelineLayout_postProcess1Tex.destroy(true);
    gPipelineLayout_postProcess0Tex.destroy(true);
    gPipelineLayout_blit2Tex.destroy(true);
    gPipelineLayout_blit1Tex.destroy(true);
    gPipelineLayout_draw_noTex.destroy(true);
    gPipelineLayout_draw.destroy(true);

    gDescSetLayout_postProcess1Tex.destroy(true);
    gDescSetLayout_postProcess0Tex.destroy(true);
    gDescSetLayout_blit3Tex.destroy(true);
    gDescSetLayout_blit2Tex.destroy(true);
    gDescSetLayout_blit1Tex.destroy(true);
    gDescSetLayout_draw_noTex.destroy(true);
    gDescSetLayout_draw.destroy(true);

    gSampler_normClampNearest.destroy();
    gSampler_draw.destroy();

    gShader_msaa_resolve_frag.destroy(true);
    gShader_gamma_adjust_post_process_frag.destroy(true);
    gShader_gamma_adjust_blit_frag.destroy(true);
    gShader_crossfade_gamma_frag.destroy(true);
    gShader_crossfade_frag.destroy(true);
    gShader_ndc_textured_frag.destroy(true);
    gShader_ndc_textured_vert.destroy(true);
    gShader_ndc_position_only_vert.destroy(true);
    gShader_sky_frag.destroy(true);
    gShader_sky_vert.destroy(true);
    gShader_world_frag.destroy(true);
    gShader_world_vert.destroy(true);
    gShader_ui_16bpp_frag.destroy(true);
    gShader_ui_8bpp_frag.destroy(true);
    gShader_ui_4bpp_frag.destroy(true);
    gShader_ui_vert.destroy(true);
    gShader_colored_frag.destroy(true);
    gShader_colored_vert.destroy(true);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Gets which pipeline set to use for the main renderpass for the current frame.
// The active set varies depending on whether gamma adjustment is in use or not, or MSAA is resolved into the swapchain image.
//------------------------------------------------------------------------------------------------------------------------------------------
const VPipelineSet<VPipelineType_Main>& getMainPipelineSet() noexcept {
    if (VRenderer::gbResolveToSwapchainThisFrame) {
        return VPipelines::gPipelines_Main_SwapchainResolve;
    } else if (VRenderer::gbUsingGammaAdjustThisFrame) {
        return VPipelines::gPipelines_Main_GammaAdjust;
    } else {
        return VPipelines::gPipelines_Main_NoGammaAdjust;
    }
}

END_NAMESPACE(VRPipelines)

#endif  // #if PSYDOOM_VULKAN_RENDERER
