#include "VRenderPath_LoadingPlaque.h"

#if PSYDOOM_VULKAN_RENDERER

#include "CmdBufferRecorder.h"
#include "LogicalDevice.h"
#include "RenderPassDef.h"
#include "Swapchain.h"

//------------------------------------------------------------------------------------------------------------------------------------------
// Sets the render path to a default uninitialized state
//------------------------------------------------------------------------------------------------------------------------------------------
VRenderPath_LoadingPlaque::VRenderPath_LoadingPlaque() noexcept
    : mbIsValid(false)
    , mpDevice(nullptr)
    , mpSwapchain(nullptr)
    , mPresentSurfaceFormat()
    , mRenderPass()
    , mFramebuffers()
    , mPreRenderPassAction()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Automatically destroys the render path if not already destroyed
//------------------------------------------------------------------------------------------------------------------------------------------
VRenderPath_LoadingPlaque::~VRenderPath_LoadingPlaque() noexcept {
    destroy();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the render path - this must always succeed
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_LoadingPlaque::init(
    vgl::LogicalDevice& device,
    vgl::Swapchain& swapchain,
    const VkFormat presentSurfaceFormat
) noexcept {
    // Sanity checks
    ASSERT_LOG(!mbIsValid, "Can't initialize twice!");
    ASSERT(device.isValid());

    // Remember all this info
    mpDevice = &device;
    mpSwapchain = &swapchain;
    mPresentSurfaceFormat = presentSurfaceFormat;

    // Create the renderpass
    if (!initRenderPass())
        FatalErrors::raise("Failed to create the loading plaque Vulkan renderpass!");

    // Now initialized
    mbIsValid = true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tears down the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_LoadingPlaque::destroy() noexcept {
    if (!mbIsValid)
        return;

    mbIsValid = false;
    mPreRenderPassAction = {};

    for (vgl::Framebuffer& framebuffer : mFramebuffers) {
        framebuffer.destroy(true);
    }

    mFramebuffers.clear();
    mRenderPass.destroy();

    mPresentSurfaceFormat = {};
    mpSwapchain = nullptr;
    mpDevice = nullptr;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates or recreates the framebuffers for this render path
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_LoadingPlaque::ensureValidFramebuffers(
    [[maybe_unused]] const uint32_t fbWidth,
    [[maybe_unused]] const uint32_t fbHeight,
    const bool bGpuIsIdle
) noexcept {
    ASSERT(mbIsValid);
    ASSERT(mpSwapchain->isValid());

    // Only do this if we need to actually create/re-create framebuffers
    if (!doFramebuffersNeedRecreate())
        return true;

    // Recreate all framebuffers
    const bool bDestroyImmediate = bGpuIsIdle;
    
    vgl::Swapchain& swapchain = *mpSwapchain;
    const uint32_t swapchainLen = swapchain.getLength();
    mFramebuffers.resize(swapchainLen);

    for (uint32_t swapImgIdx = 0; swapImgIdx < swapchainLen; ++swapImgIdx) {
        vgl::Framebuffer& framebuffer = mFramebuffers[swapImgIdx];
        framebuffer.destroy(bDestroyImmediate);

        if (!framebuffer.init(mRenderPass, swapchain, swapImgIdx, 0, {}))
            return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Begins the frame for the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_LoadingPlaque::beginFrame(vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    // Sanity checks
    ASSERT(mbIsValid);
    ASSERT(mpDevice);
    ASSERT(swapchain.isValid());
    ASSERT(swapchain.getAcquiredImageIdx() < swapchain.getVkImages().size());
    
    // Perform any actions required before the renderpass begins
    if (mPreRenderPassAction) {
        std::function<void()> action = std::move(mPreRenderPassAction);
        action();
    }

    // Begin the render pass
    const uint32_t swapchainIdx = swapchain.getAcquiredImageIdx();
    vgl::Framebuffer& framebuffer = mFramebuffers[swapchainIdx];

    cmdRec.beginRenderPass(
        mRenderPass,
        framebuffer,
        VK_SUBPASS_CONTENTS_INLINE,
        0,
        0,
        framebuffer.getWidth(),
        framebuffer.getHeight(),
        nullptr,
        0
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Ends the frame for the render path
//------------------------------------------------------------------------------------------------------------------------------------------
void VRenderPath_LoadingPlaque::endFrame([[maybe_unused]] vgl::Swapchain& swapchain, vgl::CmdBufferRecorder& cmdRec) noexcept {
    // Sanity checks and end the current render pass
    ASSERT(mbIsValid);
    ASSERT(swapchain.isValid());
    cmdRec.endRenderPass();
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Creates the Vulkan renderpass used by the render path
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_LoadingPlaque::initRenderPass() noexcept {
    // Sanity checks and getting the device
    ASSERT(mpDevice);
    vgl::LogicalDevice& device = *mpDevice;

    // Define the single color attachment
    vgl::RenderPassDef renderPassDef = {};

    VkAttachmentDescription& colorAttach = renderPassDef.attachments.emplace_back();
    colorAttach.format = mPresentSurfaceFormat;
    colorAttach.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;               // No need to clear, will be filling the screen anyway
    colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;          // Ready for presentation

    // Define the single subpass and it's attachments
    {
        vgl::SubpassDef& subpassDef = renderPassDef.subpasses.emplace_back();

        VkAttachmentReference& colorAttachRef = subpassDef.colorAttachments.emplace_back();
        colorAttachRef.attachment = 0;
        colorAttachRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // Define external subpass dependencies: these must be manually filled in
    {
        // Just block color attachment output on everything to be safe, this draw is not performance critical and outside uses are complex...
        VkSubpassDependency& dep = renderPassDef.extraSubpassDeps.emplace_back();
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = (
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT
        );
        dep.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    }
    {
        // Presentation engine reads must wait for output to be written
        VkSubpassDependency& dep = renderPassDef.extraSubpassDeps.emplace_back();
        dep.srcSubpass = 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    }

    // Finally, create the renderpass
    return mRenderPass.init(device, renderPassDef);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tells if the framebuffers need recreation
//------------------------------------------------------------------------------------------------------------------------------------------
bool VRenderPath_LoadingPlaque::doFramebuffersNeedRecreate() noexcept {
    // Firstly ensure we have the right amount of framebuffers
    ASSERT(mbIsValid);
    vgl::Swapchain& swapchain = *mpSwapchain;
    const uint32_t swapchainLen = swapchain.getLength();

    if (swapchainLen != mFramebuffers.size())
        return true;

    // Make sure each framebuffer is valid and references the correct swapchain image.
    // Also sanity check the framebuffer dimensions to ensure that they are correct.
    for (uint32_t swapImgIdx = 0; swapImgIdx < swapchainLen; ++swapImgIdx) {
        vgl::Framebuffer& framebuffer = mFramebuffers[swapImgIdx];
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

#endif  // #if PSYDOOM_VULKAN_RENDERER
