#pragma once

#if PSYDOOM_VULKAN_RENDERER

#include "Asserts.h"
#include "Macros.h"
#include "Pipeline.h"
#include "VTypes.h"

namespace vgl {
    class DescriptorSetLayout;
    class LogicalDevice;
    class Pipeline;
    class PipelineLayout;
    class RenderPass;
    class Sampler;
}

class VRenderPath_Crossfade;
class VRenderPath_LoadingPlaque;
class VRenderPath_Main;
class VRenderPath_Psx;

template <class PipelineTypeEnum>
struct VPipelineSet {
    vgl::Pipeline pipelines[(uint32_t) PipelineTypeEnum::NUM_TYPES];
    
    inline vgl::Pipeline& get(const PipelineTypeEnum pipelineId) noexcept {
        ASSERT((uint32_t) pipelineId < (uint32_t) PipelineTypeEnum::NUM_TYPES);
        return pipelines[(uint32_t) pipelineId];
    }
    
    inline const vgl::Pipeline& get(const PipelineTypeEnum pipelineId) const noexcept {
        ASSERT((uint32_t) pipelineId < (uint32_t) PipelineTypeEnum::NUM_TYPES);
        return pipelines[(uint32_t) pipelineId];
    }
};

BEGIN_NAMESPACE(VPipelines)

extern vgl::Sampler                 gSampler_draw;
extern vgl::Sampler                 gSampler_normClampNearest;
extern vgl::DescriptorSetLayout     gDescSetLayout_draw;
extern vgl::DescriptorSetLayout     gDescSetLayout_draw_noTex;
extern vgl::DescriptorSetLayout     gDescSetLayout_blit1Tex;
extern vgl::DescriptorSetLayout     gDescSetLayout_blit2Tex;
extern vgl::DescriptorSetLayout     gDescSetLayout_blit3Tex;
extern vgl::DescriptorSetLayout     gDescSetLayout_postProcess0Tex;
extern vgl::DescriptorSetLayout     gDescSetLayout_postProcess1Tex;
extern vgl::PipelineLayout          gPipelineLayout_draw;
extern vgl::PipelineLayout          gPipelineLayout_draw_noTex;
extern vgl::PipelineLayout          gPipelineLayout_blit1Tex;
extern vgl::PipelineLayout          gPipelineLayout_blit2Tex;
extern vgl::PipelineLayout          gPipelineLayout_postProcess0Tex;
extern vgl::PipelineLayout          gPipelineLayout_postProcess1Tex;
extern vgl::PipelineLayout          gPipelineLayout_crossfade;
extern vgl::PipelineLayout          gPipelineLayout_crossfadeGamma;

extern VPipelineSet<VPipelineType_Main>             gPipelines_Main_NoGammaAdjust;
extern VPipelineSet<VPipelineType_Main>             gPipelines_Main_GammaAdjust;
extern VPipelineSet<VPipelineType_Main>             gPipelines_Main_SwapchainResolve;
extern VPipelineSet<VPipelineType_Crossfade>        gPipelines_Crossfade;
extern VPipelineSet<VPipelineType_LoadingPlaque>    gPipelines_LoadingPlaque;
extern VPipelineSet<VPipelineType_PSX>              gPipelines_PSX;

void initPipelineComponents(vgl::LogicalDevice& device, const uint32_t numSamples) noexcept;

void initPipelines(
    VRenderPath_Main& mainRPath,
    VRenderPath_Psx& psxRPath,
    VRenderPath_Crossfade& crossfadeRPath,
    VRenderPath_LoadingPlaque& loadingPlaqueRPath,
    const uint32_t numSamples
) noexcept;

void shutdown() noexcept;

const VPipelineSet<VPipelineType_Main>& getMainPipelineSet() noexcept;

END_NAMESPACE(VPipelines)

#endif  // #if PSYDOOM_VULKAN_RENDERER
