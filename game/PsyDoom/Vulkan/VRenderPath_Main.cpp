#include "VRenderPath_Main.h"

#if PSYDOOM_VULKAN_RENDERER

#include "CmdBufferRecorder.h"
#include "DescriptorSet.h"
#include "LogicalDevice.h"
#include "PsyDoom/PlayerPrefs.h"
#include "RenderPassDef.h"
#include "Sampler.h"
#include "Swapchain.h"
#include "Texture.h"
#include "VDrawing.h"
#include "VPipelines.h"
#include "VRenderer.h"

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the render path to a default uninitialized state
//------------------------------------------------------------------------------------------------------------------------------------------
VRenderPath_Main::VRenderPath_Main() noexcept 
    : mbIsValid(false)
    , mpDevice(nullptr)
    , mpSwapchain(nullptr)
    , mNumDrawSamples(0)
    , mColorFormat{}
    , mResolveFormat{}
    , mPresentFormat{}
    , mRenderPass_NoGammaAdjust()
    , mRenderPass_GammaAdjust()
    , mMsaaResolver()
    , mDrawColorAttachments{}
    , mbRenderedToDrawColorAttachments{}
    , mFramebuffers_NoGammaAdjust{}
    , mFramebuffers_GammaAdjust()
    , mGammaAdjustDescriptorPool()
    , mpGammaAdjustDescriptorSets{}
    , mScreenQuad()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Automatically destroys the render path if not already destroyed
//------------------------------------------------------------------------------------------------------------------------------------------
VRenderPath_Main::~VRenderPath_Main() noexcept {
    destroy();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the render path - this must always succeed
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Main::init(
    vgl::LogicalDevice& device,
    vgl::Swapchain& swapchain,
    const uint32_t numDrawSamples,
    const VkFormat colorFormat,
    const VkFormat resolveFormat,
    const VkFormat presentFormat
) noexcept {
    // Sanity checks
    ASSERT_LOG(!mbIsValid, "Can't initialize twice!");
    ASSERT(device.isValid());

    // Remember all this info
    mpDevice = &device;
    mpSwapchain = &swapchain;
    mNumDrawSamples = numDrawSamples;
    mColorFormat = colorFormat;
    mResolveFormat = resolveFormat;
    mPresentFormat = presentFormat;

    // Create the render passes and the MSAA resolver if we are doing multi-sampling
    if (!initRenderPass(mRenderPass_NoGammaAdjust, false))
        FatalErrors::raise("Main render path: failed to create a Vulkan renderpass! (without gamma adjustment)");
        
    if (!initRenderPass(mRenderPass_GammaAdjust, true))
        FatalErrors::raise("Main render path: failed to create a Vulkan renderpass! (with gamma adjustment)");

    if (mNumDrawSamples > 1) {
        mMsaaResolver.init(device);
    }

    // Init the descriptor pool and sets used for gamma adjust
    if (!initGammaAdjustDescriptorPoolAndSets())
        FatalErrors::raise("Main render path: failed to create the gamma adjust descriptor pool and sets!");

    // Initialize the screen quad used for rendering with gamma adjustments
    mScreenQuad.init(device);

    // Now initialized
    mbIsValid = true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tears down the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Main::destroy() noexcept {
    if (!mbIsValid)
        return;

    mbIsValid = false;
    mScreenQuad.destroy(true);
    
    for (vgl::DescriptorSet*& pDescriptorSet : mpGammaAdjustDescriptorSets) {
        if (pDescriptorSet) {
            mGammaAdjustDescriptorPool.freeDescriptorSet(*pDescriptorSet);
            pDescriptorSet = nullptr;
        }
    }

    mGammaAdjustDescriptorPool.destroy(true);
    
    for (vgl::Framebuffer& framebuffer : mFramebuffers_GammaAdjust) {
        framebuffer.destroy(true);
    }
    
    mFramebuffers_GammaAdjust.clear();

    for (uint32_t i = 0; i < vgl::Defines::RINGBUFFER_SIZE; ++i) {
        mFramebuffers_NoGammaAdjust[i].destroy(true);
        mbRenderedToDrawColorAttachments[i] = false;
        mDrawColorAttachments[i].destroy(true);
    }

    mMsaaResolver.destroy();
    mRenderPass_GammaAdjust.destroy();
    mRenderPass_NoGammaAdjust.destroy();
    mResolveFormat = {};
    mColorFormat = {};
    mNumDrawSamples = 0;
    mpSwapchain = nullptr;
    mpDevice = nullptr;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates or recreates the framebuffers for this render path
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Main::ensureValidFramebuffers(
    const uint32_t fbWidth,
    const uint32_t fbHeight,
    const bool bGpuIsIdle
) noexcept {
    // Sanity checks and getting the device + swapchain
    ASSERT(mbIsValid);
    ASSERT(mpDevice);
    ASSERT(mpDevice->isValid());
    ASSERT(mpSwapchain);
    ASSERT(mpSwapchain->isValid());
    
    vgl::LogicalDevice& device = *mpDevice;
    vgl::Swapchain& swapchain = *mpSwapchain;
    
    // Recreate draw color attachments if needed
    const bool bUsingMsaa = (mNumDrawSamples > 1);
    const bool bDestroyImmediate = bGpuIsIdle;
    
    bool bRecreatedDrawColorAttachments = false;
    
    for (uint32_t i = 0; i < vgl::Defines::RINGBUFFER_SIZE; ++i) {
        vgl::RenderTexture& attachment = mDrawColorAttachments[i];
        const bool bValidAttachment = (
            attachment.isValid() &&
            (fbWidth == attachment.getWidth()) &&
            (fbHeight == attachment.getHeight())
        );
        
        if (bValidAttachment)
            continue;

        bRecreatedDrawColorAttachments = true;
        attachment.destroy(bDestroyImmediate);
        mbRenderedToDrawColorAttachments[i] = false;
        
        VkImageUsageFlags colorAttachUsage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

        if (!bUsingMsaa) {
            // Kept as the previous frame for crossfades and the loading plaque (sampled), or blitted to the swapchain
            colorAttachUsage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        } else {
            // With MSAA the samples are only ever read by the resolve subpass, in tile memory, and never stored: a transient
            // attachment, which tile based GPUs back with lazily allocated memory (no physical memory for the samples at all).
            // The frame history for crossfades etc. is the resolve attachment.
            colorAttachUsage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        }

        if (!attachment.initAsRenderTexture(device, true, mColorFormat, colorAttachUsage, fbWidth, fbHeight, mNumDrawSamples))
            return false;
    }

    // Recreate MSAA resolve attachments if needed
    bool bRecreatedMsaaResolveAttachments = false;

    if (bUsingMsaa && (!mMsaaResolver.areAllResolveAttachmentsValid(fbWidth, fbHeight))) {
        if (mMsaaResolver.createResolveAttachments(mResolveFormat, fbWidth, fbHeight)) {
            bRecreatedMsaaResolveAttachments = true;
        } else {
            return false;
        }
    }
    
    // Need to set the input attachments for the MSAA resolver after recreating the draw color attachments
    if (bUsingMsaa && bRecreatedDrawColorAttachments) {
        mMsaaResolver.setInputAttachments(mDrawColorAttachments);
    }
    
    // Check to see if all framebuffers (for the no gamma adjustment case) are valid.
    // Note: if we have recreated color or MSAA attachments then they are automatically invalid.
    bool bValidFrameBuffers_NoGammaAdjust = (
        (!bRecreatedDrawColorAttachments) &&
        (!bRecreatedMsaaResolveAttachments)
    );
    
    if (bValidFrameBuffers_NoGammaAdjust) {
        for (vgl::Framebuffer& framebuffer : mFramebuffers_NoGammaAdjust) {
            if (!framebuffer.isValid()) {
                bValidFrameBuffers_NoGammaAdjust = false;
                break;
            }
        }
    }
    
    // Check to see if all framebuffers (for the gamma adjustment case) are valid.
    // Note: if we have recreated color or MSAA attachments then they are automatically invalid.
    const uint32_t swapchainLength = swapchain.getLength();
    const uint32_t numGammaAdjustFramebuffers = swapchainLength * vgl::Defines::RINGBUFFER_SIZE;
    
    bool bValidFrameBuffers_GammaAdjust = (
        (!bRecreatedDrawColorAttachments) &&
        (!bRecreatedMsaaResolveAttachments) &&
        (numGammaAdjustFramebuffers == mFramebuffers_GammaAdjust.size())
    );
    
    if (bValidFrameBuffers_GammaAdjust) {
        // Which attachment is the swapchain attachment in the framebuffer?
        const uint32_t swapAttachmentIdx = (bUsingMsaa) ? 2u : 1u;
        
        for (uint32_t swapchainIdx = 0; swapchainIdx < swapchainLength; ++swapchainIdx) {
            for (uint32_t ringbufferIdx = 0; ringbufferIdx < vgl::Defines::RINGBUFFER_SIZE; ++ringbufferIdx) {
                const uint32_t framebufferIdx = swapchainIdx * vgl::Defines::RINGBUFFER_SIZE + ringbufferIdx;
                vgl::Framebuffer& framebuffer = mFramebuffers_GammaAdjust[framebufferIdx];
                
                bValidFrameBuffers_GammaAdjust = (
                    framebuffer.isValid() &&
                    framebuffer.isForSwapchain(swapchain) &&
                    (swapAttachmentIdx < framebuffer.getAttachmentImages().size()) &&
                    (swapchainIdx < swapchain.getLength()) &&
                    (framebuffer.getAttachmentImages()[swapAttachmentIdx] == swapchain.getVkImages()[swapchainIdx])
                );
                
                if (!bValidFrameBuffers_GammaAdjust)
                    break;
            }
            
            if (!bValidFrameBuffers_GammaAdjust)
                break;
        }
    }
    
    // Recreate the framebuffers used for the non-gamma adjusted case, if required
    if (!bValidFrameBuffers_NoGammaAdjust) {
        for (uint32_t i = 0; i < vgl::Defines::RINGBUFFER_SIZE; ++i) {
            vgl::Framebuffer& framebuffer = mFramebuffers_NoGammaAdjust[i];
            framebuffer.destroy(bDestroyImmediate);
            
            std::vector<const vgl::BaseTexture*> fbAttachments;
            fbAttachments.reserve(2);
            fbAttachments.push_back(&mDrawColorAttachments[i]);
            
            if (bUsingMsaa) {
                fbAttachments.push_back(&mMsaaResolver.getResolveAttachment(i));
            }

            if (!mFramebuffers_NoGammaAdjust[i].init(mRenderPass_NoGammaAdjust, fbAttachments))
                return false;
        }
    }

    // Recreate the framebuffers used for the gamma adjusted case, if required
    if (!bValidFrameBuffers_GammaAdjust) {
        // First tear down the original framebuffers
        for (vgl::Framebuffer& framebuffer : mFramebuffers_GammaAdjust) {
            framebuffer.destroy(bDestroyImmediate);
        }
        
        mFramebuffers_GammaAdjust.clear();
        mFramebuffers_GammaAdjust.resize(numGammaAdjustFramebuffers);
    
        // Now recreate the frame buffers
        for (uint32_t swapchainIdx = 0; swapchainIdx < swapchainLength; ++swapchainIdx) {
            for (uint32_t ringbufferIdx = 0; ringbufferIdx < vgl::Defines::RINGBUFFER_SIZE; ++ringbufferIdx) {
                const uint32_t framebufferIdx = swapchainIdx * vgl::Defines::RINGBUFFER_SIZE + ringbufferIdx;
                vgl::Framebuffer& framebuffer = mFramebuffers_GammaAdjust[framebufferIdx];
                
                std::vector<const vgl::BaseTexture*> fbAttachments;
                fbAttachments.reserve(2);
                fbAttachments.push_back(&mDrawColorAttachments[ringbufferIdx]);
                
                if (bUsingMsaa) {
                    fbAttachments.push_back(&mMsaaResolver.getResolveAttachment(ringbufferIdx));
                }
                
                const uint32_t swapImgAttachIdx = (uint32_t) fbAttachments.size(); // Index of the swapchain image attachment
                
                if (!framebuffer.init(mRenderPass_GammaAdjust, swapchain, swapchainIdx, swapImgAttachIdx, fbAttachments))
                    return false;
            }
        }
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Begins the frame for the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Main::beginFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    // Sanity checks
    ASSERT(mbIsValid);
    ASSERT(mpDevice && mpDevice->isValid());
    ASSERT(swapchain.isValid());

    // If not doing gamma adjustment we need transition the swapchain image in preparation for blitting
    const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();
    ASSERT(swapchainIdx < swapchain.getLength());

    if (!VRenderer::gbUsingGammaAdjustThisFrame) {
        VkImageMemoryBarrier imgBarrier = {};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imgBarrier.image = swapchain.getVkImages()[swapchainIdx];
        imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgBarrier.subresourceRange.levelCount = 1;
        imgBarrier.subresourceRange.layerCount = 1;

        cmdRec.addPipelineBarrier(
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            nullptr,
            1,
            &imgBarrier
        );
    }
    
    // Begin a frame for the drawing module.
    // Note: the frame might not actually be rendered/output if the swapchain becomes invalid.
    vgl::LogicalDevice& device = *mpDevice;
    const uint32_t ringbufferIdx = device.getRingbufferMgr().getBufferIndex();
    VDrawing::beginFrame(ringbufferIdx);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Ends the frame for the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Main::endFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    // Sanity checks
    ASSERT(mbIsValid);
    ASSERT(mpDevice && mpDevice->isValid());
    ASSERT(swapchain.isValid());

    // Which render pass and framebuffer are used?
    vgl::LogicalDevice& device = *mpDevice;
    const uint32_t ringbufferIdx = device.getRingbufferMgr().getBufferIndex();
    const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();

    const vgl::RenderPass& renderPass = (VRenderer::gbUsingGammaAdjustThisFrame) ?
        mRenderPass_GammaAdjust :
        mRenderPass_NoGammaAdjust;

    const vgl::Framebuffer& framebuffer = (VRenderer::gbUsingGammaAdjustThisFrame) ?
        mFramebuffers_GammaAdjust[swapchainIdx * vgl::Defines::RINGBUFFER_SIZE + ringbufferIdx] :
        mFramebuffers_NoGammaAdjust[ringbufferIdx];

    // Begin the main render pass if we are not skipping drawing.
    // Note that we don't do any drawing at all if we are skipping presenting this frame.
    // This helps avoid errors on macOS/Metal also, where we try to blit to an incompatible destination window size.
    const bool bRenderThisFrame = (!VRenderer::gbSkipNextFrame);

    if (bRenderThisFrame) {
        // Begin the render pass and clear the draw color attachment
        const VkClearValue framebufferClearValues[1] = {};

        cmdRec.beginRenderPass(
            renderPass,
            framebuffer,
            VK_SUBPASS_CONTENTS_INLINE,
            0,
            0,
            framebuffer.getWidth(),
            framebuffer.getHeight(),
            framebufferClearValues,
            C_ARRAY_SIZE(framebufferClearValues)
        );

        // Begun rendering to this framebuffer
        mbRenderedToDrawColorAttachments[ringbufferIdx] = true;
    }

    // Finish up drawing and cleanup for the frame.
    // Only record actual draw commands to the command buffer if we are NOT skipping presenting this frame.
    VDrawing::endFrame(cmdRec, bRenderThisFrame);

    if (bRenderThisFrame) {
        // Do an MSAA resolve subpass if MSAA is enabled
        const bool bUsingMsaa = (mNumDrawSamples > 1);

        if (bUsingMsaa) {
            cmdRec.nextSubpass(VK_SUBPASS_CONTENTS_INLINE);
            mMsaaResolver.resolve(cmdRec);
        }

        // If doing gamma adjust then handle that now
        if (VRenderer::gbUsingGammaAdjustThisFrame) {
            cmdRec.nextSubpass(VK_SUBPASS_CONTENTS_INLINE);
            renderGammaAdjustPostProcessingEffect(cmdRec);
        }

        // Done with the render pass now
        cmdRec.endRenderPass();

        // Make a note of what Vulkan image layout we expect the draw color attachment and MSAA resolve attachment to have
        // following the end of the render pass. This will vary depending on whether the last user was blit or a shader:
        vgl::RenderTexture& drawColorAttachment = mDrawColorAttachments[ringbufferIdx];
        
        if (bUsingMsaa || VRenderer::gbUsingGammaAdjustThisFrame) {
            drawColorAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        } else {
            drawColorAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }

        if (bUsingMsaa) {
            vgl::RenderTexture& msaaResolveAttachment = mMsaaResolver.getResolveAttachment(ringbufferIdx);

            if (VRenderer::gbUsingGammaAdjustThisFrame) {
                msaaResolveAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            } else {
                msaaResolveAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            }
        }

        // Non gamma-adjust case: blit the drawing color attachment (or MSAA resolve target, if MSAA is active) to the swapchain image.
        if (!VRenderer::gbUsingGammaAdjustThisFrame) {
            blitToSwapchainImage(swapchain, cmdRec);
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tells whether we rendered to all the draw color attachments at least once.
// This check can be used to make sure there is previous frame data for effects like plaque drawing and cross fade.
// Note: if MSAA is active this check also covers whether all MSAA resolve targets have been rendered to as well.
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Main::didRenderToAllDrawColorAttachments() noexcept {
    for (bool bRendered : mbRenderedToDrawColorAttachments) {
        if (!bRendered)
            return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the Vulkan render passes used by the render path (both gamma adjusted, and non-gamma adjusted)
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Main::initRenderPass(vgl::RenderPass& renderPass, const bool bGammaAdjusted) noexcept {
    // Sanity checks and some prerequisites
    ASSERT(mpDevice);
    vgl::LogicalDevice& device = *mpDevice;
    const bool bUsingMsaa = (mNumDrawSamples > 1);
    
    // Attachment indexes
    uint32_t mainAttachmentIdx = {};        // Main drawing attachment
    uint32_t resolveAttachmentIdx = {};     // Attachment to resolve MSAA to
    uint32_t swapchainAttachmentIdx = {};   // The swapchain attachment

    // Define the color attachment used for rendering
    vgl::RenderPassDef renderPassDef = {};

    {
        mainAttachmentIdx = (uint32_t) renderPassDef.attachments.size();
        
        VkAttachmentDescription& attach = renderPassDef.attachments.emplace_back();
        attach.format = mColorFormat;
        attach.samples = (VkSampleCountFlagBits) mNumDrawSamples;   // Can just cast for the correct conversion
        attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // When using MSAA or gamma adjustment, then this attachment will be drawn via a shader and used as an intermediate input.
        // Because of this, we don't care about storing it's contents to VRAM at the end of the render pass, and thus can save on memory bandwidth.
        // If NOT using MSAA or gamma adjustment then we blit this attachment to the swapchain image (outside of the render pass), in which case we must store it to VRAM first.
        if (bUsingMsaa) {
            attach.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;              // Don't need to write to VRAM at the end of the renderpass (image unused outside of the renderpass).
            attach.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // Don't care: would say undefined but that is not allowed.
        }
        else if (bGammaAdjusted) {
            // Stored: crossfades and the loading plaque sample the previous frames from this attachment (no MSAA). With
            // DONT_CARE a tile based GPU really drops it, so those effects showed undefined contents with gamma adjustment.
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attach.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        else {
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;              // Needs to be stored to VRAM for the blit (which happens outside of the render pass).
            attach.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // Ready for blitting to the swapchain image.
        }
    }

    // If doing MSAA, define the MSAA color resolve attachment
    if (bUsingMsaa) {
        resolveAttachmentIdx = (uint32_t) renderPassDef.attachments.size();
        
        VkAttachmentDescription& attach = renderPassDef.attachments.emplace_back();
        attach.format = mResolveFormat;
        attach.samples = VK_SAMPLE_COUNT_1_BIT;
        attach.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        
        if (bGammaAdjusted) {
            // Stored: the previous frames for crossfades and the loading plaque (MSAA) are read from this attachment
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attach.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        else {
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;              // Needs to be stored to VRAM for the blit (which happens outside of the render pass).
            attach.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // Ready for blitting to the swapchain image
        }
    }
    
    // If doing gamma adjustment then define the gamma adjustment attachment (the swapchain image)
    if (bGammaAdjusted) {
        swapchainAttachmentIdx = (uint32_t) renderPassDef.attachments.size();
        
        VkAttachmentDescription& attach = renderPassDef.attachments.emplace_back();
        attach.format = mPresentFormat;
        attach.samples = VK_SAMPLE_COUNT_1_BIT;
        attach.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // Make it ready for present
    }

    // Define the main 'draw' subpass and it's attachments
    {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();

        VkAttachmentReference& dstAttach = subpassDef.colorAttachments.emplace_back();
        dstAttach.attachment = mainAttachmentIdx;
        dstAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // If doing MSAA, define the MSAA resolve subpass and it's attachments
    if (bUsingMsaa) {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();

        VkAttachmentReference& srcAttach = subpassDef.inputAttachments.emplace_back();
        srcAttach.attachment = mainAttachmentIdx;
        srcAttach.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference& dstAttach = subpassDef.colorAttachments.emplace_back();
        dstAttach.attachment = resolveAttachmentIdx;
        dstAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    
    // If doing gamma adjustment define the gamma adjust subpass and it's attachments
    if (bGammaAdjusted) {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();
        
        VkAttachmentReference& srcAttach = subpassDef.inputAttachments.emplace_back();
        srcAttach.attachment = (bUsingMsaa) ? resolveAttachmentIdx : mainAttachmentIdx;
        srcAttach.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        
        VkAttachmentReference& dstAttach = subpassDef.colorAttachments.emplace_back();
        dstAttach.attachment = swapchainAttachmentIdx;
        dstAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // Define external subpass dependencies - these must be manually filled in.
    {
        // Outside of the current render pass (in a previous frame), the main draw attachment may have been used for:
        //  (1) Blitting to the swapchain image (No MSAA, no gamma adjustment)
        //  (2) Drawing to the swapchain image (No MSAA, gamma adjustment active)
        //  (3) Resolving to the MSAA resolve attachment (MSAA active, gamma adjustment active or inactive)
        //
        // To be safe, wait for all of those operations to be completed.
        // With gamma adjustment in particular, we don't know what settings the previous frame may have used, so handle all cases.
        VkSubpassDependency& dep = renderPassDef.extraSubpassDeps.emplace_back();
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }

    // When we are not doing gamma adjustment then a blit will be used for final output.
    // Since this happens outside of the render pass, we must add extra synchronization here.
    if (!bGammaAdjusted) {
        // Blit must wait on drawing or msaa resolve to finish (depending on whether msaa is enabled)
        VkSubpassDependency& dep = renderPassDef.extraSubpassDeps.emplace_back();
        dep.srcSubpass = (bUsingMsaa) ? 1 : 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    }

    // Finally, create the renderpass.
    // Note: this will automatically determine a lot of the subpass dependencies inside the render pass.
    return renderPass.init(device, renderPassDef);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the descriptor pool and descriptor sets used for gamma adjustment
//-----------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Main::initGammaAdjustDescriptorPoolAndSets() noexcept {
    ASSERT(mpDevice);
    ASSERT(mpDevice->isValid());

    // Make the descriptor pool
    std::vector<VkDescriptorPoolSize> poolResourceCounts;
    poolResourceCounts.resize(2);
    
    poolResourceCounts[0].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolResourceCounts[0].descriptorCount = vgl::Defines::RINGBUFFER_SIZE;
    poolResourceCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolResourceCounts[1].descriptorCount = vgl::Defines::RINGBUFFER_SIZE;

    if (!mGammaAdjustDescriptorPool.init(*mpDevice, poolResourceCounts, vgl::Defines::RINGBUFFER_SIZE))
        return false;

    // Alloc the descriptor sets (these will be filled in fully later)
    for (vgl::DescriptorSet*& pDescriptorSet : mpGammaAdjustDescriptorSets) {
        pDescriptorSet = mGammaAdjustDescriptorPool.allocDescriptorSet(VPipelines::gDescSetLayout_postProcess1Tex);

        if (!pDescriptorSet)
            return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Renders a screen quad using the gamma adjustment post-process shader.
// Expected to be called during the gamma adjust subpass.
//-----------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Main::renderGammaAdjustPostProcessingEffect(vgl::CmdBufferRecorder& cmdRec) noexcept {
    vgl::LogicalDevice& device = *mpDevice;
    
    // Doing gamma adjust: bind the resources used for it to the gamma adjust descriptor set for this frame
    ASSERT(VRenderer::gGammaAdjustTex.isValid());
    
    const uint32_t ringbufferIdx = device.getRingbufferMgr().getBufferIndex();
    const VkSampler gammaAdjustSampler = VPipelines::gSampler_normClampNearest.getVkSampler();
    const bool bUsingMsaa = (mNumDrawSamples > 1);
    
    vgl::DescriptorSet& gammaAdjustDescSet = *mpGammaAdjustDescriptorSets[ringbufferIdx];
    vgl::RenderTexture& inputAttachment = (bUsingMsaa) ? mMsaaResolver.getResolveAttachment(ringbufferIdx) : mDrawColorAttachments[ringbufferIdx];
    
    VkDescriptorImageInfo imageInfos[2] = {};
    imageInfos[0].sampler = gammaAdjustSampler;
    imageInfos[0].imageView = inputAttachment.getVkImageView();
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfos[1].sampler = gammaAdjustSampler;
    imageInfos[1].imageView = VRenderer::gGammaAdjustTex.getVkImageView();
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    gammaAdjustDescSet.bindInputAttachments(0, &imageInfos[0], 1);
    gammaAdjustDescSet.bindTextures(1, &imageInfos[1], 1);
    
    // Render a screen quad using the gamma adjust shader
    vgl::Pipeline& drawPipeline = VPipelines::gPipelines_Main_GammaAdjust.get(VPipelineType_Main::GammaAdjustPostProcess);
    
    mScreenQuad.bindVertexBuffer(cmdRec, 0);
    cmdRec.bindPipeline(drawPipeline);
    cmdRec.bindDescriptorSet(*mpGammaAdjustDescriptorSets[ringbufferIdx], drawPipeline, 0);
    mScreenQuad.draw(cmdRec);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Blits the final output image to the swapchain image.
// The final output image can either be the draw color attachment (no MSAA) or the MSAA resolve attachment (MSAA active).
// This blit is used for the case where we are NOT doing any gamma adjustment.
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Main::blitToSwapchainImage(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    ASSERT(mpDevice && mpDevice->isValid());
    ASSERT(swapchain.isValid());

    vgl::LogicalDevice& device = *mpDevice;
    const uint32_t ringbufferIdx = device.getRingbufferMgr().getBufferIndex();
    const vgl::Framebuffer& framebuffer = mFramebuffers_NoGammaAdjust[ringbufferIdx];
    const bool bUsingMsaa = (mNumDrawSamples > 1);
    
    const VkImage blitSrcImage = (bUsingMsaa) ?
        mMsaaResolver.getResolveAttachment(ringbufferIdx).getVkImage() :    // Blit from the MSAA resolve color buffer
        framebuffer.getAttachmentImages()[0];                               // No MSAA: blit directly from the color buffer that was drawn to

    const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();
    const VkImage swapchainImage = swapchain.getVkImages()[swapchainIdx];

    // Note that we must first wait for writes to the image to finish up from the render pass.
    // Hence add an image barrier first:
    {
        VkImageMemoryBarrier imgBarrier = {};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        imgBarrier.image = blitSrcImage;
        imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgBarrier.subresourceRange.levelCount = 1;
        imgBarrier.subresourceRange.layerCount = 1;

        cmdRec.addPipelineBarrier(
            VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            nullptr,
            1,
            &imgBarrier
        );
    }

    {
        VkImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.srcOffsets[1].x = framebuffer.getWidth();
        blitRegion.srcOffsets[1].y = framebuffer.getHeight();
        blitRegion.srcOffsets[1].z = 1;
        blitRegion.dstOffsets[1].x = swapchain.getSwapExtentWidth();
        blitRegion.dstOffsets[1].y = swapchain.getSwapExtentHeight();
        blitRegion.dstOffsets[1].z = 1;

        cmdRec.blitImage(
            blitSrcImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchainImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blitRegion,
            VK_FILTER_NEAREST
        );
    }

    // Transition the swapchain image back to presentation optimal in preparation for presentation
    {
        VkImageMemoryBarrier imgBarrier = {};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        imgBarrier.image = swapchainImage;
        imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgBarrier.subresourceRange.levelCount = 1;
        imgBarrier.subresourceRange.layerCount = 1;

        cmdRec.addPipelineBarrier(
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            nullptr,
            1,
            &imgBarrier
        );
    }
}
#endif  // #if PSYDOOM_VULKAN_RENDERER
