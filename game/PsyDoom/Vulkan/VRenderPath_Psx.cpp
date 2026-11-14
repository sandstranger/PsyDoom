#include "VRenderPath_Psx.h"

#if PSYDOOM_VULKAN_RENDERER

#include "Asserts.h"
#include "CmdBufferRecorder.h"
#include "DescriptorSet.h"
#include "Gpu.h"
#include "LogicalDevice.h"
#include "Pipeline.h"
#include "PsyDoom/PsxVm.h"
#include "PsyDoom/Video.h"
#include "RenderPassDef.h"
#include "Sampler.h"
#include "Swapchain.h"
#include "VDrawing.h"
#include "VPipelines.h"
#include "VRenderer.h"

#include <algorithm>
#include <cmath>

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the render path to a default uninitialized state
//------------------------------------------------------------------------------------------------------------------------------------------
VRenderPath_Psx::VRenderPath_Psx() noexcept 
    : mbIsValid(false)
    , mpDevice(nullptr)
    , mpSwapchain(nullptr)
    , mRenderPass()
    , mPsxFramebufferTextures{}
    , mGammaAdjustFramebuffers()
    , mGammaAdjustVerts()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Automatically destroys the render path if not already destroyed
//------------------------------------------------------------------------------------------------------------------------------------------
VRenderPath_Psx::~VRenderPath_Psx() noexcept {
    destroy();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the render path - this must always succeed
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Psx::init(
    vgl::LogicalDevice& device,
    vgl::Swapchain& swapchain,
    const VkFormat psxFramebufferFormat,
    const VkFormat presentSurfaceFormat
) noexcept {
    // Sanity checks and remembering the device used
    ASSERT_LOG(!mbIsValid, "Can't initialize twice!");
    ASSERT(device.isValid());

    mpDevice = &device;
    mpSwapchain = &swapchain;

    // Only two framebuffer formats are supported
    ASSERT((psxFramebufferFormat == VK_FORMAT_A1R5G5B5_UNORM_PACK16) || (psxFramebufferFormat == VK_FORMAT_B8G8R8A8_UNORM));

    // Create the renderpass used for gamma adjustment
    if (!initGammaAdjustRenderPass(presentSurfaceFormat))
        FatalErrors::raise("PSX render path: failed to create the gamma adjust renderpass!");
        
    // Init the descriptor pool and sets used for gamma adjust
    if (!initGammaAdjustDescriptorPoolAndSets())
        FatalErrors::raise("PSX render path: failed to create the gamma adjust descriptor pool and sets!");

    // Initialize the PSX framebuffer textures used to hold the old PSX renderer framebuffer before it is blit to the Vulkan framebuffer.
    // Note: enable these textures to be read/copied from also!
    for (uint32_t i = 0; i < vgl::Defines::RINGBUFFER_SIZE; ++i) {
        vgl::Texture& psxFbTex = mPsxFramebufferTextures[i];
        
        if (!psxFbTex.initAs2dTexture(device, psxFramebufferFormat, Video::ORIG_DRAW_RES_X, Video::ORIG_DRAW_RES_Y, true))  // N.B: 'true' for copyable!
            FatalErrors::raise("PSX render path: failed to create a Vulkan texture for the PSX framebuffer!");
    }
    
    // Initialize the vertex buffer sets used for doing gamma adjust
    mGammaAdjustVerts.init<VVertex_XyUv>(device, 6);

    // Now initialized
    mbIsValid = true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tears down the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Psx::destroy() noexcept {
    if (!mbIsValid)
        return;

    mbIsValid = false;
    mGammaAdjustVerts.destroy();
    
    for (vgl::DescriptorSet*& pDescriptorSet : mpGammaAdjustDescriptorSets) {
        if (pDescriptorSet) {
            mGammaAdjustDescriptorPool.freeDescriptorSet(*pDescriptorSet);
            pDescriptorSet = nullptr;
        }
    }

    mGammaAdjustDescriptorPool.destroy(true);
    
    for (vgl::Framebuffer& framebuffer : mGammaAdjustFramebuffers) {
        framebuffer.destroy(true);
    }
    
    mGammaAdjustFramebuffers.clear();

    for (vgl::Texture& texture : mPsxFramebufferTextures) {
        texture.destroy(true);
    }

    mRenderPass.destroy();

    mpSwapchain = nullptr;
    mpDevice = nullptr;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates or recreates the gamma adjust framebuffers for this render path
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Psx::ensureValidFramebuffers(
    [[maybe_unused]] const uint32_t fbWidth,
    [[maybe_unused]] const uint32_t fbHeight,
    const bool bGpuIsIdle
) noexcept {
    ASSERT(mbIsValid);
    ASSERT(mpSwapchain->isValid());

    // Only do this if we need to actually create/re-create the gamma adjust framebuffers
    if (!doGammaAdjustFramebuffersNeedRecreate())
        return true;

    // Recreate all gamma adjust framebuffers
    const bool bDestroyImmediate = bGpuIsIdle;
    
    vgl::Swapchain& swapchain = *mpSwapchain;
    const uint32_t swapchainLen = swapchain.getLength();
    mGammaAdjustFramebuffers.resize(swapchainLen);

    for (uint32_t swapImgIdx = 0; swapImgIdx < swapchainLen; ++swapImgIdx) {
        vgl::Framebuffer& framebuffer = mGammaAdjustFramebuffers[swapImgIdx];
        framebuffer.destroy(bDestroyImmediate);

        if (!framebuffer.init(mRenderPass, swapchain, swapImgIdx, 0, {}))
            return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Begins the frame for the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Psx::beginFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    // Sanity checks
    ASSERT(mbIsValid);
    ASSERT(mpDevice);
    ASSERT(mpDevice->isValid());
    ASSERT(swapchain.isValid());
    ASSERT(swapchain.getAcquiredImageIdx() < swapchain.getLength());
    ASSERT(swapchain.getAcquiredImageIdx() < mGammaAdjustFramebuffers.size());

    // Are we doing a gamma adjust? If so then the frame must be handled differently.
    // Note that for gamma adjust we'll do all the work in 'endFrame()'
    if (!VRenderer::gbUsingGammaAdjustThisFrame) {
        // Not doing any gamma adjust. PSX framebuffer will be blitted to the swapchain image directly.
        // Transition the swapchain image to transfer destination optimal in preparation for blitting.
        const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();
        ASSERT(swapchainIdx < swapchain.getLength());

        const VkImage swapchainImage = swapchain.getVkImages()[swapchainIdx];

        {
            VkImageMemoryBarrier imgBarrier = {};
            imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imgBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            imgBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
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

        // Clear the swapchain image to opaque black.
        // This is needed for the classic PSX renderer as not all parts of the image might be filled.
        {
            VkClearColorValue clearColor = {};
            clearColor.float32[3] = 1.0f;

            VkImageSubresourceRange imageResRange = {};
            imageResRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageResRange.levelCount = 1;
            imageResRange.layerCount = 1;

            cmdRec.clearColorImage(swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &imageResRange);
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Ends the frame for the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Psx::endFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    // Sanity checks
    ASSERT(mbIsValid);
    ASSERT(swapchain.isValid());
    ASSERT(swapchain.getAcquiredImageIdx() < swapchain.getLength());

    // Copy the PlayStation 1 framebuffer to the current framebuffer texture
    vgl::LogicalDevice& device = *mpDevice;
    const uint32_t ringbufferIdx = device.getRingbufferMgr().getBufferIndex();
    vgl::Texture& psxFbTexture = mPsxFramebufferTextures[ringbufferIdx];

    {
        Gpu::Core& gpu = PsxVm::gGpu;
        const uint16_t* pSrcPixels = gpu.pRam + (gpu.displayAreaX + (uintptr_t) gpu.displayAreaY * gpu.ramPixelW);
        const std::byte* const pDstTextureBytes = psxFbTexture.lock();

        if (psxFbTexture.getFormat() == VK_FORMAT_A1R5G5B5_UNORM_PACK16) {
            copyPsxFramebufferToFbTexture_A1R5G5B5(pSrcPixels, (uint16_t*) pDstTextureBytes);
        } else {
            ASSERT(psxFbTexture.getFormat() == VK_FORMAT_B8G8R8A8_UNORM);
            copyPsxFramebufferToFbTexture_B8G8R8A8(pSrcPixels, (uint32_t*) pDstTextureBytes);
        }

        psxFbTexture.unlock();
    }

    // Only bother doing further commands if we're going to present.
    // This avoids errors on MacOS/Metal also, where we try to blit to an incompatible destination window size.
    if (VRenderer::gbSkipNextFrame || VRenderer::gbSkipNextFramePresent)
        return;

    // Get the area of the window to blit the PSX framebuffer to
    const uint32_t screenWidth = swapchain.getSwapExtentWidth();
    const uint32_t screenHeight = swapchain.getSwapExtentHeight();

    float blitDstX_f = {};
    float blitDstY_f = {};
    float blitDstW_f = {};
    float blitDstH_f = {};
    Video::getClassicFramebufferWindowRect((float) screenWidth, (float) screenHeight, blitDstX_f, blitDstY_f, blitDstW_f, blitDstH_f);
    
    const int32_t blitDstMinX = std::clamp<int32_t>((int32_t) blitDstX_f, 0, screenWidth);
    const int32_t blitDstMinY = std::clamp<int32_t>((int32_t) blitDstY_f, 0, screenHeight);
    const int32_t blitDstMaxX = std::clamp<int32_t>((int32_t)(blitDstX_f + std::ceil(blitDstW_f)), 0, screenWidth);
    const int32_t blitDstMaxY = std::clamp<int32_t>((int32_t)(blitDstY_f + std::ceil(blitDstH_f)), 0, screenHeight);
    const int32_t blitW = blitDstMaxX - blitDstMinX;
    const int32_t blitH = blitDstMaxY - blitDstMinY;
    
    // Wait for uploads to finish then transition the PSX framebuffer for rendering.
    // If we are blitting directly (no gamma adjust) then it becomes transfer source optimal.
    // If we are doing gamma adjust then it must be shader read-only optimal.
    const bool bDoingGammaAdjust = VRenderer::gbUsingGammaAdjustThisFrame;
    
    {
        VkImageMemoryBarrier imgBarrier = {};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        
        if (bDoingGammaAdjust) {
            imgBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else {
            imgBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }
        
        imgBarrier.image = psxFbTexture.getVkImage();
        imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgBarrier.subresourceRange.levelCount = 1;
        imgBarrier.subresourceRange.layerCount = 1;
        
        const VkPipelineStageFlagBits waitPipelineStages = (bDoingGammaAdjust) ?
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT :
            VK_PIPELINE_STAGE_TRANSFER_BIT;

        cmdRec.addPipelineBarrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            waitPipelineStages,
            0,
            nullptr,
            1,
            &imgBarrier
        );
        
        psxFbTexture.setVkImageLayoutHint(imgBarrier.newLayout);
    }
    
    // Are we doing gamma adjust?
    // The method of finishing up the frame differs greatly if so.
    if (bDoingGammaAdjust) {
        // Doing gamma adjust: bind the resources used for it to the gamma adjust descriptor set for this frame.
        // Note: using 2 separate bindings instead of a texture array to appease MoltenVK on macOS.
        ASSERT(VRenderer::gGammaAdjustTex.isValid());
        
        const VkSampler gammaAdjustSampler = VPipelines::gSampler_normClampNearest.getVkSampler();
        vgl::DescriptorSet& gammaAdjustDescSet = *mpGammaAdjustDescriptorSets[ringbufferIdx];
        
        VkDescriptorImageInfo imageInfos[2] = {};
        imageInfos[0].sampler = gammaAdjustSampler;
        imageInfos[0].imageView = mPsxFramebufferTextures[ringbufferIdx].getVkImageView();
        imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos[1].sampler = gammaAdjustSampler;
        imageInfos[1].imageView = VRenderer::gGammaAdjustTex.getVkImageView();
        imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        gammaAdjustDescSet.bindTextures(0, &imageInfos[0], 1);
        gammaAdjustDescSet.bindTextures(1, &imageInfos[1], 1);
    
        // Render the PSX framebuffer to the swapchain using the gamma adjust shader.
        // Begin the render pass and clear the swapchain image.
        const uint32_t swapchainImageIdx = swapchain.getAcquiredImageIdx();
        vgl::Framebuffer& framebuffer = mGammaAdjustFramebuffers[swapchainImageIdx];
        VkClearValue framebufferClearValues[1] = {};

        cmdRec.beginRenderPass(
            mRenderPass,
            framebuffer,
            VK_SUBPASS_CONTENTS_INLINE,
            0,
            0,
            framebuffer.getWidth(),
            framebuffer.getHeight(),
            framebufferClearValues,
            C_ARRAY_SIZE(framebufferClearValues)
        );
        
        // Alloc and fill in the vertices for rendering gamma adjust
        mGammaAdjustVerts.beginFrame(ringbufferIdx);
        VVertex_XyUv* const pVerts = mGammaAdjustVerts.allocVerts<VVertex_XyUv>(6);
        
        const float lx = ((float) blitDstMinX / (float) screenWidth - 0.5f) * 2.0f;
        const float rx = ((float) blitDstMaxX / (float) screenWidth - 0.5f) * 2.0f;
        const float ty = ((float) blitDstMinY / (float) screenHeight - 0.5f) * 2.0f;
        const float by = ((float) blitDstMaxY / (float) screenHeight - 0.5f) * 2.0f;
        
        const float tv = (float) Video::gTopOverscan / (float) Video::ORIG_DRAW_RES_Y;
        const float bv = (float)(Video::ORIG_DRAW_RES_Y - Video::gBotOverscan) / (float) Video::ORIG_DRAW_RES_Y;
        
        pVerts[0] = VVertex_XyUv{ lx, ty, 0.0f, tv };
        pVerts[1] = VVertex_XyUv{ rx, ty, 1.0f, tv };
        pVerts[2] = VVertex_XyUv{ rx, by, 1.0f, bv };
        pVerts[3] = VVertex_XyUv{ rx, by, 1.0f, bv };
        pVerts[4] = VVertex_XyUv{ lx, by, 0.0f, bv };
        pVerts[5] = VVertex_XyUv{ lx, ty, 0.0f, tv };
        
        // Draw the PSX framebuffer to the swapchain with gamma adjustment
        vgl::Pipeline& drawPipeline = VPipelines::gPipelines_PSX.get(VPipelineType_PSX::GammaAdjustBlit);
        
        VRenderer::setupViewportAndScissors(cmdRec);
        cmdRec.bindVertexBuffer(mGammaAdjustVerts.buffers[ringbufferIdx], 0, 0);
        cmdRec.bindPipeline(drawPipeline);
        cmdRec.bindDescriptorSet(*mpGammaAdjustDescriptorSets[ringbufferIdx], drawPipeline, 0);
        cmdRec.draw(6, 0);
        
        // Finish up drawing to the vertex buffer and the render pass
        mGammaAdjustVerts.endFrame();
        cmdRec.endRenderPass();
    }
    else {
        // Not doing gamma adjust: blit the PSX framebuffer directly to the swapchain image
        ASSERT((Video::gTopOverscan >= 0) && (Video::gTopOverscan < Video::ORIG_DRAW_RES_Y / 2));
        ASSERT((Video::gBotOverscan >= 0) && (Video::gBotOverscan < Video::ORIG_DRAW_RES_Y / 2));

        const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();
        const VkImage vkSwapchainImage = swapchain.getVkImages()[swapchainIdx];

        if ((blitW > 0) && (blitH > 0)) {
            // Ensure the clear of the swapchain image is finished before we blit
            {
                VkImageMemoryBarrier imgBarrier = {};
                imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                imgBarrier.image = vkSwapchainImage;
                imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                imgBarrier.subresourceRange.levelCount = 1;
                imgBarrier.subresourceRange.layerCount = 1;

                cmdRec.addPipelineBarrier(
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    nullptr,
                    1,
                    &imgBarrier
                );
            }

            // Do the blit to the swapchain image
            {
                VkImageBlit blitRegion = {};
                blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blitRegion.srcSubresource.layerCount = 1;
                blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blitRegion.dstSubresource.layerCount = 1;
                blitRegion.srcOffsets[0].y = Video::gTopOverscan;
                blitRegion.srcOffsets[1].x = Video::ORIG_DRAW_RES_X;
                blitRegion.srcOffsets[1].y = Video::ORIG_DRAW_RES_Y - Video::gBotOverscan;
                blitRegion.srcOffsets[1].z = 1;
                blitRegion.dstOffsets[0].x = blitDstMinX;
                blitRegion.dstOffsets[0].y = blitDstMinY;
                blitRegion.dstOffsets[1].x = blitDstMaxX;
                blitRegion.dstOffsets[1].y = blitDstMaxY;
                blitRegion.dstOffsets[1].z = 1;

                cmdRec.blitImage(
                    psxFbTexture.getVkImage(),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    vkSwapchainImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &blitRegion,
                    VK_FILTER_NEAREST
                );
            }
        }

        // Transition the swapchain image back to presentation optimal in preparation for presentation
        {
            VkImageMemoryBarrier imgBarrier = {};
            imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imgBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            imgBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            imgBarrier.image = vkSwapchainImage;
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
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the Vulkan render pass used for gamma adjust
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Psx::initGammaAdjustRenderPass(const VkFormat presentSurfaceFormat) noexcept {
    // Sanity checks and some prerequisites
    ASSERT(mpDevice);
    vgl::LogicalDevice& device = *mpDevice;

    // Define the output color attachment (the swapchain image)
    vgl::RenderPassDef renderPassDef = {};

    VkAttachmentDescription& colorAttach = renderPassDef.attachments.emplace_back();
    colorAttach.format = presentSurfaceFormat;
    colorAttach.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Define the single subpass and it's attachments
    {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();
        VkAttachmentReference& colorAttachRef = subpassDef.colorAttachments.emplace_back();
        colorAttachRef.attachment = 0;
        colorAttachRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // Define external subpass dependencies: these must be manually filled in.
    {
        // The draw attachment must wait for any previous drawing or blits to end
        VkSubpassDependency& dep = renderPassDef.extraSubpassDeps.emplace_back();
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }

    // Finally, create the renderpass
    return mRenderPass.init(device, renderPassDef);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the descriptor pool and descriptor sets used for gamma adjustment
//-----------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Psx::initGammaAdjustDescriptorPoolAndSets() noexcept {
    ASSERT(mpDevice);
    ASSERT(mpDevice->isValid());

    // Make the descriptor pool
    VkDescriptorPoolSize poolResourceCount = {};
    poolResourceCount.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolResourceCount.descriptorCount = 2u * vgl::Defines::RINGBUFFER_SIZE;

    if (!mGammaAdjustDescriptorPool.init(*mpDevice, { poolResourceCount }, vgl::Defines::RINGBUFFER_SIZE))
        return false;

    // Alloc the descriptor sets (these will be filled in fully later)
    for (vgl::DescriptorSet*& pDescriptorSet : mpGammaAdjustDescriptorSets) {
        pDescriptorSet = mGammaAdjustDescriptorPool.allocDescriptorSet(VPipelines::gDescSetLayout_blit2Tex);

        if (!pDescriptorSet)
            return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tells if the framebuffers used for gamma adjust need recreation
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_Psx::doGammaAdjustFramebuffersNeedRecreate() noexcept {
    // Firstly ensure we have the right amount of framebuffers
    ASSERT(mbIsValid);
    vgl::Swapchain& swapchain = *mpSwapchain;
    const uint32_t swapchainLen = swapchain.getLength();

    if (swapchainLen != mGammaAdjustFramebuffers.size())
        return true;

    // Make sure each framebuffer is valid and references the correct swapchain image.
    // Also sanity check the framebuffer dimensions to ensure that they are correct.
    for (uint32_t swapImgIdx = 0; swapImgIdx < swapchainLen; ++swapImgIdx) {
        vgl::Framebuffer& framebuffer = mGammaAdjustFramebuffers[swapImgIdx];
        const std::vector<VkImage>& framebufferImages = framebuffer.getAttachmentImages();
        const std::vector<VkImage>& swapchainImages = swapchain.getVkImages();
        
        const bool bValidFramebuffer = (
            framebuffer.isValid() &&
            framebuffer.isForSwapchain(swapchain) &&
            (framebufferImages.size() == 1) &&
            (framebufferImages[0] == swapchainImages[swapImgIdx]) &&
            (framebuffer.getWidth() == swapchain.getSwapExtentWidth()) &&
            (framebuffer.getHeight() == swapchain.getSwapExtentHeight())
        );

        if (!bValidFramebuffer)
            return true;
    }

    // If we get to here then the framebuffers don't need recreation
    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Copies the PSX framebuffer to a Vulkan texture containing the same framebuffer.
// This overload is for an A1R5G5B5 format destination framebuffer.
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Psx::copyPsxFramebufferToFbTexture_A1R5G5B5(const uint16_t* const pSrcPixels, uint16_t* pDstPixels) noexcept {
    Gpu::Core& gpu = PsxVm::gGpu;
    const uint16_t ramPixelW = gpu.ramPixelW;

    const uint16_t* pSrcRowPixels = pSrcPixels;
    uint16_t* pDstRowPixels = pDstPixels;

    for (uint32_t y = 0; y < Video::ORIG_DRAW_RES_Y; ++y) {
        // Note: don't bother doing multiple pixels at a time - compiler is smart and already optimizes this to use SIMD
        for (uint32_t x = 0; x < Video::ORIG_DRAW_RES_X; ++x) {
            const uint16_t srcPixel = pSrcRowPixels[x];
            const uint16_t srcR = (srcPixel >>  0) & 0x1F;
            const uint16_t srcG = (srcPixel >>  5) & 0x1F;
            const uint16_t srcB = (srcPixel >> 10) & 0x1F;

            const uint16_t dstPixel = (srcR << 10) | (srcG << 5) | (srcB << 0) | 0x8000;
            pDstRowPixels[x] = dstPixel;
        }

        pSrcRowPixels += ramPixelW;
        pDstRowPixels += Video::ORIG_DRAW_RES_X;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Copies the PSX framebuffer to a Vulkan texture containing the same framebuffer.
// This overload is for an B8G8R8A8 format destination framebuffer.
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_Psx::copyPsxFramebufferToFbTexture_B8G8R8A8(const uint16_t* const pSrcPixels, uint32_t* pDstPixels) noexcept {
    Gpu::Core& gpu = PsxVm::gGpu;
    const uint16_t ramPixelW = gpu.ramPixelW;

    const uint16_t* pSrcRowPixels = pSrcPixels;
    uint32_t* pDstRowPixels = pDstPixels;

    for (uint32_t y = 0; y < Video::ORIG_DRAW_RES_Y; ++y) {
        // Note: don't bother doing multiple pixels at a time - compiler is smart and already optimizes this to use SIMD
        for (uint32_t x = 0; x < Video::ORIG_DRAW_RES_X; ++x) {
            const uint32_t srcPixel = pSrcRowPixels[x];
            const uint32_t srcR = ((srcPixel >>  0) & 0x1F) << 3;
            const uint32_t srcG = ((srcPixel >>  5) & 0x1F) << 3;
            const uint32_t srcB = ((srcPixel >> 10) & 0x1F) << 3;

            const uint32_t dstPixel = (srcB << 0) | (srcG << 8) | (srcR << 16) | 0xFF000000u;
            pDstRowPixels[x] = dstPixel;
        }

        pSrcRowPixels += ramPixelW;
        pDstRowPixels += Video::ORIG_DRAW_RES_X;
    }
}

#endif  // #if PSYDOOM_VULKAN_RENDERER
