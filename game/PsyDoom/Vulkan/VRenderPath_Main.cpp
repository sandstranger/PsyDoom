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
    , mRenderPass_SwapchainResolve()
    , mbHardwareMsaaResolve(false)
    , mMsaaResolver()
    , mDrawColorAttachments{}
    , mbRenderedToDrawColorAttachments{}
    , mFramebuffers_NoGammaAdjust{}
    , mFramebuffers_GammaAdjust()
    , mFramebuffers_SwapchainResolve()
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

    // Tile based GPUs (Android): MSAA is resolved by the hardware as the tiles are written out (pResolveAttachments), so the samples
    // are the only thing in tile memory: 4x MSAA of 32-bit color is 128 bits per pixel, which Mali (Bifrost, Valhall up to G57/G7x)
    // holds at full tile size. A resolve subpass with a shader kept the samples, the resolve attachment and the swapchain image in
    // the tile at once (192 bits at 4x: smaller tiles). A hardware resolve needs the format of the samples for the resolve target;
    // the other builds may draw in 16-bit color and resolve to 32-bit, which only the shader can do, so they keep it.
#if ANDROID
    mbHardwareMsaaResolve = ((numDrawSamples > 1) && (colorFormat == resolveFormat));
#else
    mbHardwareMsaaResolve = false;
#endif

    // Create the render passes and the MSAA resolver if we are doing multi-sampling
    if (!initRenderPass(mRenderPass_NoGammaAdjust, false, false))
        FatalErrors::raise("Main render path: failed to create a Vulkan renderpass! (without gamma adjustment)");

    if (!initRenderPass(mRenderPass_GammaAdjust, true, false))
        FatalErrors::raise("Main render path: failed to create a Vulkan renderpass! (with gamma adjustment)");

    // Hardware resolve straight into the swapchain image too, which must have the format of the samples
    if (mbHardwareMsaaResolve && (colorFormat == presentFormat)) {
        if (!initRenderPass(mRenderPass_SwapchainResolve, false, true))
            FatalErrors::raise("Main render path: failed to create a Vulkan renderpass! (MSAA resolved into the swapchain image)");
    }

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

    for (vgl::Framebuffer& framebuffer : mFramebuffers_SwapchainResolve) {
        framebuffer.destroy(true);
    }

    mFramebuffers_SwapchainResolve.clear();

    for (uint32_t i = 0; i < vgl::Defines::RINGBUFFER_SIZE; ++i) {
        mFramebuffers_NoGammaAdjust[i].destroy(true);
        mbRenderedToDrawColorAttachments[i] = false;
        mDrawColorAttachments[i].destroy(true);
    }

    mMsaaResolver.destroy();
    mRenderPass_SwapchainResolve.destroy();
    mRenderPass_GammaAdjust.destroy();
    mRenderPass_NoGammaAdjust.destroy();
    mbHardwareMsaaResolve = false;
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
        
        VkImageUsageFlags colorAttachUsage = 0;

        if (!bUsingMsaa) {
            // Kept as the previous frame for crossfades and the loading plaque (sampled), or blitted to the swapchain.
            // Read by the gamma adjust subpass as an input attachment.
            colorAttachUsage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        } else if (mbHardwareMsaaResolve) {
            // With MSAA the samples never leave tile memory: a transient attachment, which tile based GPUs back with lazily allocated
            // memory (no physical memory for the samples at all). The hardware resolves them as the tiles are written out, it is only
            // ever a color attachment. The frame history for crossfades etc. is the resolve attachment.
            colorAttachUsage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        } else {
            // The same, but the samples are read by the shader of the resolve subpass, in tile memory
            colorAttachUsage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
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
    
    // Need to set the input attachments for the MSAA resolver after recreating the draw color attachments (its shader resolves them)
    if (bUsingMsaa && (!mbHardwareMsaaResolve) && bRecreatedDrawColorAttachments) {
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

    // Recreate the framebuffers of the render passes writing the swapchain image (gamma adjust, MSAA resolved into it), if required.
    // Note: if we have recreated color or MSAA attachments then they are automatically invalid.
    const bool bRecreatedAttachments = (bRecreatedDrawColorAttachments || bRecreatedMsaaResolveAttachments);

    if (!initSwapchainFramebuffers(mFramebuffers_GammaAdjust, mRenderPass_GammaAdjust, swapchain, bRecreatedAttachments, bDestroyImmediate))
        return false;

    if (canResolveToSwapchain()) {
        if (!initSwapchainFramebuffers(mFramebuffers_SwapchainResolve, mRenderPass_SwapchainResolve, swapchain, bRecreatedAttachments, bDestroyImmediate))
            return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Makes sure the framebuffers of a render pass writing the swapchain image are valid, recreating them if required.
// There is one per swapchain image and ringbuffer slot, arranged as [SwapchainSize][RINGBUFFER_SIZE]. The attachments are the draw
// color attachment, the MSAA resolve attachment (if doing MSAA) and the swapchain image.
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Main::initSwapchainFramebuffers(
    std::vector<vgl::Framebuffer>& framebuffers,
    const vgl::RenderPass& renderPass,
    vgl::Swapchain& swapchain,
    const bool bForceRecreate,
    const bool bDestroyImmediate
) noexcept {
    const bool bUsingMsaa = (mNumDrawSamples > 1);
    const uint32_t swapchainLength = swapchain.getLength();
    const uint32_t numFramebuffers = swapchainLength * vgl::Defines::RINGBUFFER_SIZE;
    const uint32_t swapAttachmentIdx = (bUsingMsaa) ? 2u : 1u;  // Which attachment is the swapchain image

    // Are the existing framebuffers still valid?
    bool bValidFramebuffers = ((!bForceRecreate) && (framebuffers.size() == numFramebuffers));

    for (uint32_t swapchainIdx = 0; bValidFramebuffers && (swapchainIdx < swapchainLength); ++swapchainIdx) {
        for (uint32_t ringbufferIdx = 0; bValidFramebuffers && (ringbufferIdx < vgl::Defines::RINGBUFFER_SIZE); ++ringbufferIdx) {
            const vgl::Framebuffer& framebuffer = framebuffers[swapchainIdx * vgl::Defines::RINGBUFFER_SIZE + ringbufferIdx];

            bValidFramebuffers = (
                framebuffer.isValid() &&
                framebuffer.isForSwapchain(swapchain) &&
                (swapAttachmentIdx < framebuffer.getAttachmentImages().size()) &&
                (framebuffer.getAttachmentImages()[swapAttachmentIdx] == swapchain.getVkImages()[swapchainIdx])
            );
        }
    }

    if (bValidFramebuffers)
        return true;

    // First tear down the original framebuffers
    for (vgl::Framebuffer& framebuffer : framebuffers) {
        framebuffer.destroy(bDestroyImmediate);
    }

    framebuffers.clear();
    framebuffers.resize(numFramebuffers);

    // Now recreate them
    for (uint32_t swapchainIdx = 0; swapchainIdx < swapchainLength; ++swapchainIdx) {
        for (uint32_t ringbufferIdx = 0; ringbufferIdx < vgl::Defines::RINGBUFFER_SIZE; ++ringbufferIdx) {
            vgl::Framebuffer& framebuffer = framebuffers[swapchainIdx * vgl::Defines::RINGBUFFER_SIZE + ringbufferIdx];

            std::vector<const vgl::BaseTexture*> fbAttachments;
            fbAttachments.reserve(2);
            fbAttachments.push_back(&mDrawColorAttachments[ringbufferIdx]);

            if (bUsingMsaa) {
                fbAttachments.push_back(&mMsaaResolver.getResolveAttachment(ringbufferIdx));
            }

            const uint32_t swapImgAttachIdx = (uint32_t) fbAttachments.size();  // Index of the swapchain image attachment

            if (!framebuffer.init(renderPass, swapchain, swapchainIdx, swapImgAttachIdx, fbAttachments))
                return false;
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

    // If the render pass doesn't write the swapchain image (gamma adjust, MSAA resolved into it) we need transition the swapchain image
    // in preparation for blitting.
    const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();
    ASSERT(swapchainIdx < swapchain.getLength());

    if ((!VRenderer::gbUsingGammaAdjustThisFrame) && (!VRenderer::gbResolveToSwapchainThisFrame)) {
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

    // MSAA resolved by the hardware into the frame history and the swapchain image, gamma adjusted into the swapchain image, or blitted to it
    const bool bResolveToSwapchain = VRenderer::gbResolveToSwapchainThisFrame;
    const bool bGammaAdjust = VRenderer::gbUsingGammaAdjustThisFrame;
    const uint32_t swapchainFramebufferIdx = swapchainIdx * vgl::Defines::RINGBUFFER_SIZE + ringbufferIdx;
    ASSERT((!bResolveToSwapchain) || (canResolveToSwapchain() && (!bGammaAdjust)));

    const vgl::RenderPass& renderPass = (bResolveToSwapchain) ? mRenderPass_SwapchainResolve :
        (bGammaAdjust) ? mRenderPass_GammaAdjust : mRenderPass_NoGammaAdjust;

    const vgl::Framebuffer& framebuffer = (bResolveToSwapchain) ? mFramebuffers_SwapchainResolve[swapchainFramebufferIdx] :
        (bGammaAdjust) ? mFramebuffers_GammaAdjust[swapchainFramebufferIdx] : mFramebuffers_NoGammaAdjust[ringbufferIdx];

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
        // Do an MSAA resolve subpass if MSAA is enabled and resolved by a shader.
        // A hardware resolve happens at the end of the draw subpass, into the frame history.
        const bool bUsingMsaa = (mNumDrawSamples > 1);

        if (bUsingMsaa && (!mbHardwareMsaaResolve)) {
            cmdRec.nextSubpass(VK_SUBPASS_CONTENTS_INLINE);
            mMsaaResolver.resolve(cmdRec);
        }

        if (bResolveToSwapchain) {
            // Nothing is drawn in this subpass: at its end the hardware resolves the same samples once more, into the swapchain image
            cmdRec.nextSubpass(VK_SUBPASS_CONTENTS_INLINE);
        }
        else if (bGammaAdjust) {
            // If doing gamma adjust then handle that now
            cmdRec.nextSubpass(VK_SUBPASS_CONTENTS_INLINE);
            renderGammaAdjustPostProcessingEffect(cmdRec);
        }

        // Done with the render pass now
        cmdRec.endRenderPass();

        // Make a note of what Vulkan image layout we expect the draw color attachment and MSAA resolve attachment to have
        // following the end of the render pass. This will vary depending on whether the last user was blit or a shader:
        vgl::RenderTexture& drawColorAttachment = mDrawColorAttachments[ringbufferIdx];

        if (bUsingMsaa && mbHardwareMsaaResolve) {
            drawColorAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);     // Only ever a color attachment
        } else if (bUsingMsaa || bGammaAdjust) {
            drawColorAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        } else {
            drawColorAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }

        if (bUsingMsaa) {
            vgl::RenderTexture& msaaResolveAttachment = mMsaaResolver.getResolveAttachment(ringbufferIdx);

            if (bGammaAdjust || bResolveToSwapchain) {
                msaaResolveAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            } else {
                msaaResolveAttachment.setVkImageLayoutHint(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            }
        }

        // Neither case writing the swapchain image: blit the drawing color attachment (or MSAA resolve target, if MSAA is active) to it
        if ((!bGammaAdjust) && (!bResolveToSwapchain)) {
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
// Creates the Vulkan render passes used by the render path (both gamma adjusted, and non-gamma adjusted).
//
// With hardware MSAA resolve, 'bResolveToSwapchain' makes the render pass for presenting straight from tile memory: the draw subpass
// resolves the samples into the frame history (the resolve attachment, kept for crossfades and the loading plaque), then an empty
// subpass with the same color attachment resolves them once more, into the swapchain image. Both resolves happen as the tiles are
// written out, so the samples are the only thing in tile memory.
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Main::initRenderPass(vgl::RenderPass& renderPass, const bool bGammaAdjusted, const bool bResolveToSwapchain) noexcept {
    // Sanity checks and some prerequisites
    ASSERT(mpDevice);
    ASSERT((!bResolveToSwapchain) || (mbHardwareMsaaResolve && (!bGammaAdjusted) && (mColorFormat == mPresentFormat)));
    vgl::LogicalDevice& device = *mpDevice;
    const bool bUsingMsaa = (mNumDrawSamples > 1);
    const bool bHardwareMsaaResolve = (bUsingMsaa && mbHardwareMsaaResolve);
    const bool bShaderMsaaResolve = (bUsingMsaa && (!mbHardwareMsaaResolve));
    const bool bWritesSwapchain = (bGammaAdjusted || bResolveToSwapchain);

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

            // Don't care: would say undefined but that is not allowed. A hardware resolved attachment is only ever a color attachment.
            attach.finalLayout = (bHardwareMsaaResolve) ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
        
        if (bWritesSwapchain) {
            // Stored: the previous frames for crossfades and the loading plaque (MSAA) are read from this attachment
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attach.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        else {
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;              // Needs to be stored to VRAM for the blit (which happens outside of the render pass).
            attach.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // Ready for blitting to the swapchain image
        }
    }
    
    // If doing gamma adjustment (or resolving MSAA into it) then define the gamma adjustment attachment (the swapchain image)
    if (bWritesSwapchain) {
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

        // Hardware MSAA resolve: into the frame history, at the end of this subpass (as the tiles are written out)
        if (bHardwareMsaaResolve) {
            VkAttachmentReference& resolveAttach = subpassDef.colorResolveAttachments.emplace_back();
            resolveAttach.attachment = resolveAttachmentIdx;
            resolveAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }

    // If doing MSAA resolved by a shader, define the MSAA resolve subpass and it's attachments
    if (bShaderMsaaResolve) {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();

        VkAttachmentReference& srcAttach = subpassDef.inputAttachments.emplace_back();
        srcAttach.attachment = mainAttachmentIdx;
        srcAttach.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference& dstAttach = subpassDef.colorAttachments.emplace_back();
        dstAttach.attachment = resolveAttachmentIdx;
        dstAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    
    // If resolving MSAA into the swapchain image: a subpass without drawing whose color attachment is the same multisampled one, so
    // the hardware resolves the samples again at its end, into the swapchain image. The subpass dependency on the draw subpass is a
    // 'by region' one: a tile based GPU merges the two subpasses and does both resolves when the tiles are written out.
    if (bResolveToSwapchain) {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();

        VkAttachmentReference& colorAttach = subpassDef.colorAttachments.emplace_back();
        colorAttach.attachment = mainAttachmentIdx;
        colorAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference& resolveAttach = subpassDef.colorResolveAttachments.emplace_back();
        resolveAttach.attachment = swapchainAttachmentIdx;
        resolveAttach.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    // When the render pass doesn't write the swapchain image then a blit will be used for final output.
    // Since this happens outside of the render pass, we must add extra synchronization here.
    if (!bWritesSwapchain) {
        // Blit must wait on drawing or msaa resolve to finish (depending on whether msaa is enabled and how it is resolved)
        VkSubpassDependency& dep = renderPassDef.extraSubpassDeps.emplace_back();
        dep.srcSubpass = (bShaderMsaaResolve) ? 1 : 0;
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
