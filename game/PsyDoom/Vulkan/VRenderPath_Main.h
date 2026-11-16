#pragma once

#if PSYDOOM_VULKAN_RENDERER

#include "Asserts.h"
#include "DescriptorPool.h"
#include "Framebuffer.h"
#include "IVRenderPath.h"
#include "RenderPass.h"
#include "VMsaaResolver.h"
#include "VScreenQuad.h"

//------------------------------------------------------------------------------------------------------------------------------------------
// This is the primary renderer path for the new Vulkan renderer.
// All normal drawing operations are done via this render path.
//------------------------------------------------------------------------------------------------------------------------------------------
class VRenderPath_Main : public IVRendererPath {
public:
    VRenderPath_Main() noexcept;
    ~VRenderPath_Main() noexcept;

    void init(
        vgl::LogicalDevice& device,
        vgl::Swapchain& swapchain,
        const uint32_t numDrawSamples,
        const VkFormat colorFormat,
        const VkFormat resolveFormat,
        const VkFormat presentFormat
    ) noexcept;

    void destroy() noexcept;
    
    virtual bool ensureValidFramebuffers(
        const uint32_t fbWidth,
        const uint32_t fbHeight,
        const bool bGpuIsIdle
    ) noexcept override;
    
    virtual void beginFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept override;
    virtual void endFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept override;

    inline bool isValid() const noexcept { return mbIsValid; }
    inline uint32_t getNumDrawSamples() const noexcept { return mNumDrawSamples; }
    inline const vgl::RenderPass& getRenderPass_NoGammaAdjust() const noexcept { return mRenderPass_NoGammaAdjust; }
    inline const vgl::RenderPass& getRenderPass_GammaAdjust() const noexcept { return mRenderPass_GammaAdjust; }
    inline const vgl::RenderPass& getRenderPass_SwapchainResolve() const noexcept { return mRenderPass_SwapchainResolve; }
    inline VMsaaResolver& getMsaaResolver() noexcept { return mMsaaResolver; }

    // MSAA resolved by the hardware as the tiles are written out (pResolveAttachments), instead of by a shader in a subpass of its own
    inline bool usesHardwareMsaaResolve() const noexcept { return mbHardwareMsaaResolve; }

    // With hardware MSAA resolve: whether the samples can also be resolved straight into the swapchain image (same format)
    inline bool canResolveToSwapchain() const noexcept { return mRenderPass_SwapchainResolve.isValid(); }

    inline vgl::RenderTexture& getDrawColorAttachment(const uint32_t ringbufferIdx) noexcept {
        ASSERT(ringbufferIdx < vgl::Defines::RINGBUFFER_SIZE);
        return mDrawColorAttachments[ringbufferIdx];
    }

    bool didRenderToAllDrawColorAttachments() noexcept;

private:
    bool initRenderPass(vgl::RenderPass& renderPass, const bool bGammaAdjusted, const bool bResolveToSwapchain) noexcept;
    bool initSwapchainFramebuffers(
        std::vector<vgl::Framebuffer>& framebuffers,
        const vgl::RenderPass& renderPass,
        vgl::Swapchain& swapchain,
        const bool bForceRecreate,
        const bool bDestroyImmediate
    ) noexcept;
    bool initGammaAdjustDescriptorPoolAndSets() noexcept;
    void renderGammaAdjustPostProcessingEffect(vgl::CmdBufferRecorder& cmdRec) noexcept;
    void blitToSwapchainImage(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept;

    bool                    mbIsValid;                  // True if the render path has been initialized
    vgl::LogicalDevice*     mpDevice;                   // The Vulkan device used
    vgl::Swapchain*         mpSwapchain;                // The swapchain that the game uses for display
    uint32_t                mNumDrawSamples;            // How many samples to use during drawing, '1' if MSAA is disabled
    VkFormat                mColorFormat;               // The format used for the main/rendering color attachment
    VkFormat                mResolveFormat;             // If doing MSAA this is the format to use for the MSAA resolve target
    VkFormat                mPresentFormat;             // The format used by the swapchain
    vgl::RenderPass         mRenderPass_NoGammaAdjust;  // The Vulkan render pass for this render path (non-gamma adjusted case)
    vgl::RenderPass         mRenderPass_GammaAdjust;    // The Vulkan render pass for this render path (gamma adjusted case)
    vgl::RenderPass         mRenderPass_SwapchainResolve;   // Hardware MSAA resolve only: the samples resolved into the frame history and the swapchain image
    bool                    mbHardwareMsaaResolve;      // MSAA is resolved by the hardware as the tiles are written out, not by the shader of the resolver
    VMsaaResolver           mMsaaResolver;              // Only initialized if doing MSAA: helper to help resolve the multi-sampled framebuffer

    // The color attachments used for all drawing: one per ringbuffer slot.
    // If MSAA is active these can be multi-sampled attachments.
    vgl::RenderTexture mDrawColorAttachments[vgl::Defines::RINGBUFFER_SIZE];
    
    // Whether each of the color attachments used for drawing have been involved in a frame yet.
    // Note: if MSAA is active then each array entry also implies that the corresponding MSAA resolve target has been drawn to.
    bool mbRenderedToDrawColorAttachments[vgl::Defines::RINGBUFFER_SIZE];

    // Framebuffers for when gamma adjustment is not being used: one per ringbuffer slot
    vgl::Framebuffer mFramebuffers_NoGammaAdjust[vgl::Defines::RINGBUFFER_SIZE];
    
    // Framebuffers for when gamma adjustment is active.
    // The number of entries in this array is SwapchainSize x RINGBUFFER_SIZE.
    // In terms of a 2d array, the framebuffers are arranged as [SwapchainSize][RINGBUFFER_SIZE].
    std::vector<vgl::Framebuffer> mFramebuffers_GammaAdjust;

    // Framebuffers for 'mRenderPass_SwapchainResolve', arranged the same way as the gamma adjust ones
    std::vector<vgl::Framebuffer> mFramebuffers_SwapchainResolve;

    // Used for gamma adjustment: a descriptor pool and the descriptor sets allocated from it.
    // The descriptor sets just contain a binding for an input multi-sampled attachment to resolve - one for each ringbuffer index so we don't have to update constantly.
    vgl::DescriptorPool mGammaAdjustDescriptorPool;
    vgl::DescriptorSet* mpGammaAdjustDescriptorSets[vgl::Defines::RINGBUFFER_SIZE];
    
    // A screen quad used for rendering the gamma adjustment effect
    VScreenQuad mScreenQuad;
};

#endif  // #if PSYDOOM_VULKAN_RENDERER
