//------------------------------------------------------------------------------------------------------------------------------------------
// A module that handles the submission of draw commands to the Vulkan renderer.
// Allows various primitives to be submitted.
// Records the vertex data into a buffer and the commands into the current command buffer used by the renderer.
//------------------------------------------------------------------------------------------------------------------------------------------
#include "VDrawing.h"

#if PSYDOOM_VULKAN_RENDERER

#include "Asserts.h"
#include "Buffer.h"
#include "CmdBufferRecorder.h"
#include "Defines.h"
#include "DescriptorPool.h"
#include "DescriptorSet.h"
#include "Doom/doomdef.h"
#include "FatalErrors.h"
#include "Framebuffer.h"
#include "Gpu.h"
#include "LogicalDevice.h"
#include "Pipeline.h"
#include "PsyDoom/Config/Config.h"
#include "PsyDoom/Video.h"
#include "VPipelines.h"
#include "VRenderer.h"
#include "VTypes.h"
#include "VVertexBufferSet.h"

#include <algorithm>
#include <cstring>
#include <memory>

BEGIN_NAMESPACE(VDrawing)

// Ringbuffer index for the current frame being generated
static uint32_t gCurRingbufferIdx;

// Descriptor set and a descriptor pool used for all 'draw' subpass operations.
// Binds the PSX VRAM texture to it's combined image sampler in binding 0.
static vgl::DescriptorPool  gDescriptorPool;
static vgl::DescriptorSet*  gpDescriptorSet;

// Vertex buffers: for the 'draw' subpass (VVertex_Draw)
static VVertexBufferSet gVertexBuffers_Draw;

// The current pipeline being used by the 'draw' subpass; used to help avoid unneccessary pipeline switches
static VPipelineType_Main gCurDrawPipelineType;

// Sets of uniforms for the current frame
static std::vector<VShaderUniforms_Draw> gFrameUniforms;

// Drawing commands for the current frame
static std::vector<DrawCmd> gFrameDrawCmds;

#if PSYDOOM_GLES_RENDERER
    //--------------------------------------------------------------------------------------------------------------------------------------
    // GLES renderer: vertices for the 'draw' subpass go into CPU memory (uploaded once at the end of the frame).
    // Has the same batch tracking as 'VVertexBufferSet'.
    //--------------------------------------------------------------------------------------------------------------------------------------
    struct CpuVertexBuffer {
        std::unique_ptr<VVertex_Draw[]>     pVerts;
        uint32_t                            capacity;
        uint32_t                            curOffset;
        uint32_t                            curBatchStart;
        uint32_t                            curBatchSize;
        bool                                bInFrame;

        void init(const uint32_t initialCapacity) noexcept {
            pVerts = std::make_unique<VVertex_Draw[]>(initialCapacity);
            capacity = initialCapacity;
            curOffset = 0;
            curBatchStart = 0;
            curBatchSize = 0;
            bInFrame = false;
        }

        void destroy() noexcept {
            pVerts.reset();
            capacity = 0;
            curOffset = 0;
            curBatchStart = 0;
            curBatchSize = 0;
            bInFrame = false;
        }

        VVertex_Draw* allocVerts(const uint32_t numVerts) noexcept {
            if (!bInFrame)
                FatalErrors::raise("VDrawing: attempting to draw outside of a GLES renderer frame!");

            if (curOffset + numVerts > capacity) {
                uint32_t newCapacity = std::max(capacity, 1024u);

                while (curOffset + numVerts > newCapacity) {
                    newCapacity *= 2;
                }

                std::unique_ptr<VVertex_Draw[]> pNewVerts = std::make_unique<VVertex_Draw[]>(newCapacity);
                std::memcpy(pNewVerts.get(), pVerts.get(), (size_t) curOffset * sizeof(VVertex_Draw));
                pVerts = std::move(pNewVerts);
                capacity = newCapacity;
            }

            VVertex_Draw* const pAllocVerts = pVerts.get() + curOffset;
            curOffset += numVerts;
            curBatchSize += numVerts;
            return pAllocVerts;
        }

        void endCurrentDrawBatch() noexcept {
            if (curBatchSize > 0) {
                curBatchStart = curOffset;
                curBatchSize = 0;
            }
        }
    };

    static bool             gbUseCpuVertexBuffer;   // True when the GLES renderer uses this module
    static CpuVertexBuffer  gCpuVertexBuffer;
#endif

//------------------------------------------------------------------------------------------------------------------------------------------
// Allocates vertices for the 'draw' subpass in the current draw batch: in the Vulkan vertex buffer, or in CPU memory for GLES
//------------------------------------------------------------------------------------------------------------------------------------------
static inline VVertex_Draw* allocDrawVerts(const uint32_t numVerts) noexcept {
    #if PSYDOOM_GLES_RENDERER
        if (gbUseCpuVertexBuffer)
            return gCpuVertexBuffer.allocVerts(numVerts);
    #endif

    return gVertexBuffers_Draw.allocVerts<VVertex_Draw>(numVerts);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tells if the specified 'Main' pipeline type uses a descriptor set or not.
// This will be false for pipelines that don't use textures.
//------------------------------------------------------------------------------------------------------------------------------------------
static bool doesPipelineTypeUseDescriptorSet(const VPipelineType_Main pipelineType) noexcept {
    return (
        (pipelineType != VPipelineType_Main::Lines) &&
        (pipelineType != VPipelineType_Main::Colored)
    );
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Records all drawing commands for the current frame to a Vulkan command buffer
//------------------------------------------------------------------------------------------------------------------------------------------
static void recordCmdBuffer(vgl::CmdBufferRecorder& cmdRec) noexcept {
    // First command in the drawing pass is to configure the viewport and scissors settings
    VRenderer::setupViewportAndScissors(cmdRec);

    // Bind the correct vertex buffer for drawing (unless there is nothing to draw)
    if (!gFrameDrawCmds.empty()) {
        ASSERT(gVertexBuffers_Draw.pCurBuffer);
        cmdRec.bindVertexBuffer(*gVertexBuffers_Draw.pCurBuffer, 0, 0);
    }

    // Clear this flag once we bind the drawing descriptor set - it only needs to be done once since all draw pipeline layouts are compatible
    bool bNeedToBindDescriptorSet = true;

    // Decide which set of pipelines are being used for drawing
    const VPipelineSet<VPipelineType_Main>& pipelineSet = VPipelines::getMainPipelineSet();

    // Handle each draw command
    for (const DrawCmd& drawCmd : gFrameDrawCmds) {
        switch (drawCmd.type) {
            case DrawCmdType::SetPipeline: {
                // Bind the pipeline
                const VPipelineType_Main pipelineType = (VPipelineType_Main) drawCmd.arg1;
                const vgl::Pipeline& pipeline = pipelineSet.get(pipelineType);
                cmdRec.bindPipeline(pipeline);

                // Do we need to bind the draw descriptor set after binding the pipeline?
                // Note that some pipelines don't use the draw descriptor set, such as those for line and solid color drawing.
                // Binding those types of pipeline also invalidates any bound descriptor set, due to the incompatible pipeline layout.
                const bool bUsingDescriptorSet = doesPipelineTypeUseDescriptorSet(pipelineType);
                
                if (bUsingDescriptorSet) {
                    if (bNeedToBindDescriptorSet) {
                        cmdRec.bindDescriptorSet(*gpDescriptorSet, pipeline, 0, 0, nullptr);
                        bNeedToBindDescriptorSet = false;
                    }
                }
                else {
                    bNeedToBindDescriptorSet = true;
                }
            }   break;

            case DrawCmdType::SetUniforms: {
                cmdRec.pushConstants(
                    VPipelines::gPipelineLayout_draw,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(VShaderUniforms_Draw),
                    &gFrameUniforms[drawCmd.arg1]
                );
            }   break;

            case DrawCmdType::Draw: {
                cmdRec.draw(drawCmd.arg1, drawCmd.arg2);
            }   break;
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Initializes the drawing module and allocates draw vertex buffers etc.
//------------------------------------------------------------------------------------------------------------------------------------------
void init(vgl::LogicalDevice& device, vgl::BaseTexture& vramTex) noexcept {
    // Create the descriptor pool and descriptor set used for rendering.
    // Only using a single descriptor set which is bound to the PSX VRAM texture.
    {
        // Make the descriptor pool
        VkDescriptorPoolSize poolResources[1] = {};
        poolResources[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolResources[0].descriptorCount = vgl::Defines::RINGBUFFER_SIZE;

        if (!gDescriptorPool.init(device, { poolResources[0] }, 1))
            FatalErrors::raise("VDrawing: Failed to create a Vulkan descriptor pool!");

        // Make the descriptor set and bind the PSX VRAM texture and sampler to slot 0.
        // This particular binding only needs to be done once at startup!
        gpDescriptorSet = gDescriptorPool.allocDescriptorSet(VPipelines::gDescSetLayout_draw);

        if (!gpDescriptorSet)
            FatalErrors::raise("VDrawing: Failed to allocate a required Vulkan descriptor set!");

        gpDescriptorSet->bindTextureAndSampler(0, vramTex, VPipelines::gSampler_draw);
    }

    // Create the vertex buffers
    constexpr uint32_t DRAW_VB_SIZE = 4 * 1024 * 1024;
    gVertexBuffers_Draw.init<VVertex_Draw>(device, DRAW_VB_SIZE / sizeof(VVertex_Draw));

    // Current draw pipeline in use is undefined initially
    gCurDrawPipelineType = (VPipelineType_Main) -1;

    // Prealloc draw buffer memory
    gFrameUniforms.reserve(16);
    gFrameDrawCmds.reserve(4196);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Shuts down the drawing module and frees up resources
//------------------------------------------------------------------------------------------------------------------------------------------
void shutdown() noexcept {
    gFrameDrawCmds.clear();
    gFrameUniforms.clear();
    gCurDrawPipelineType = {};

    #if PSYDOOM_GLES_RENDERER
        if (gbUseCpuVertexBuffer) {
            gCpuVertexBuffer.destroy();
            gbUseCpuVertexBuffer = false;
            return;
        }
    #endif

    gVertexBuffers_Draw.destroy();

    if (gpDescriptorSet) {
        gpDescriptorSet->free(true);
        gpDescriptorSet = nullptr;
    }

    gDescriptorPool.destroy(true);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Performs start of frame logic for the drawing module
//------------------------------------------------------------------------------------------------------------------------------------------
void beginFrame(const uint32_t ringbufferIdx) noexcept {
    // Remember which ringbuffer slot we are on and setup vertex buffers for the frame
    gCurRingbufferIdx = ringbufferIdx;
    gVertexBuffers_Draw.beginFrame(ringbufferIdx);

    // Expect all this to already have the following state
    ASSERT(gCurDrawPipelineType == (VPipelineType_Main) -1);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Performs end of frame logic for the drawing module.
// Optionally, draw commands can be skipped if they are not required - if we are not outputting a frame, for example.
//------------------------------------------------------------------------------------------------------------------------------------------
void endFrame(vgl::CmdBufferRecorder& cmdRec, const bool bRenderThisFrame) noexcept {
    // Finish the current draw batch then record all drawing commands in the Vulkan command buffer (unless skipping)
    endCurrentDrawBatch();
    
    if (bRenderThisFrame) {
        recordCmdBuffer(cmdRec);
    }

    // Upload vertices generated during drawing, so the draw commands can use them
    gVertexBuffers_Draw.endFrame();

    // Post frame cleanup: clear buffers, the current draw pipeline and ringbuffer index
    gFrameDrawCmds.clear();
    gFrameUniforms.clear();
    gCurDrawPipelineType = (VPipelineType_Main) -1;
    gCurRingbufferIdx = {};
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Tells if any draw commands have been submitted
//------------------------------------------------------------------------------------------------------------------------------------------
bool hasIssuedDrawCmds() noexcept {
    return (!gFrameDrawCmds.empty());
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Set which pipeline is being used for the 'draw' subpass with lazy early out if there is no change
//------------------------------------------------------------------------------------------------------------------------------------------
void setDrawPipeline(const VPipelineType_Main type) noexcept {
    // Only switch pipelines if we need to
    ASSERT((uint32_t) type < (uint32_t) VPipelineType_Main::NUM_TYPES);
    const VPipelineType_Main oldPipelineType = gCurDrawPipelineType;

    if (oldPipelineType == type)
        return;

    // Do the pipeline switch and record the switch draw command.
    // Note that we must end the current draw batch before switching!
    endCurrentDrawBatch();
    gCurDrawPipelineType = type;

    DrawCmd& drawCmd = gFrameDrawCmds.emplace_back();
    drawCmd.type = DrawCmdType::SetPipeline;
    drawCmd.arg1 = (uint32_t) type;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Set uniforms used by draw shaders, including the model/view/projection transform matrix
//------------------------------------------------------------------------------------------------------------------------------------------
void setDrawUniforms(const VShaderUniforms_Draw& uniforms) noexcept {
    // Record the command to set the uniforms and save them for later
    DrawCmd& drawCmd = gFrameDrawCmds.emplace_back();
    drawCmd.type = DrawCmdType::SetUniforms;
    drawCmd.arg1 = (uint32_t) gFrameUniforms.size();

    gFrameUniforms.emplace_back(uniforms);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Compute the transform matrix for the UI/2D elements in Doom.
// Scales and positions everything such that the original UI coordinates can be used for the same result.
//------------------------------------------------------------------------------------------------------------------------------------------
Matrix4f computeTransformMatrixForUI(const bool bAllowWidescreen) noexcept {
    // The UI is centered horizontally in the viewport (if we are in widescreen mode) and occupies it's full vertical range.
    // Compute how much leftover space there would be at the left and right due to widescreen.
    const float xPadding = (VRenderer::gPsxCoordsFbX / VRenderer::gPsxCoordsFbW) * (float) SCREEN_W;

    // These are the projection parameters.
    // Note: need to reverse by/ty to get the view the right way round (normally y is up with projection matrices).
    const float viewLx = (bAllowWidescreen) ? -xPadding : 0.0f;
    const float viewRx = (bAllowWidescreen) ? (float) SCREEN_W + xPadding : (float) SCREEN_W;
    const float viewTy = (float) Video::gTopOverscan;
    const float viewBy = (float) SCREEN_H - (float) Video::gBotOverscan;
    const float zNear = 0.0f;
    const float zFar = 1.0f;

    return Matrix4f::orthographicOffCenter(viewLx, viewRx, viewTy, viewBy, zNear, zFar);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Compute the transform matrix used to setup the 3D view for PsyDoom
//------------------------------------------------------------------------------------------------------------------------------------------
Matrix4f computeTransformMatrixFor3D(const float viewX, const float viewY, const float viewZ, const float viewAngle) noexcept {
    // Projection parameters.
    // Project as if we still had the old 256x200 3D viewport and off center to account for the status bar at the bottom.
    // This matches the projection that the original PSX Doom used, with allowance for widescreen:
    constexpr float STATUS_BAR_H = SCREEN_H - VIEW_3D_H;

    const bool bAllowWidescreen = Config::gbVulkanWidescreenEnabled;
    const float widescreenScale = std::max((float) VRenderer::gFramebufferW / (float) VRenderer::gPsxCoordsFbW, 1.0f);
    const float viewLx = (bAllowWidescreen) ? -widescreenScale : -1.0f;
    const float viewRx = (bAllowWidescreen) ? +widescreenScale : +1.0f;
    const float viewTy = (HALF_VIEW_3D_H - (float) Video::gTopOverscan) / (float) HALF_SCREEN_W;
    const float viewBy = (-HALF_VIEW_3D_H - STATUS_BAR_H + (float) Video::gBotOverscan) / (float) HALF_SCREEN_W;
    const float zNear = 1.0f;
    const float zFar = 65536.0f;

    // Compute the projection, rotation and translation matrices for the view.
    // Rotation is about the Y axis (Doom's Z axis).
    const Matrix4f projection = Matrix4f::perspectiveOffCenter(viewLx, viewRx, viewTy, viewBy, zNear, zFar);
    const Matrix4f rotation = Matrix4f::rotateY(viewAngle);
    const Matrix4f translation = Matrix4f::translate(-viewX, -viewY, -viewZ);

    // Now combine all the matrices and return the result
    const Matrix4f modelViewProj = translation * rotation * projection;
    return modelViewProj;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Ends the current primitive batch for the 'draw' subpass and issues the command to draw the primitives.
// The primitives are drawn with whatever pipeline is currently bound.
//------------------------------------------------------------------------------------------------------------------------------------------
void endCurrentDrawBatch() noexcept {
    // Records the draw command for the current batch of the given vertex storage, if there are vertices in it
    const auto endBatch = [](auto& vertexStorage) noexcept {
        if (vertexStorage.curBatchSize <= 0)
            return;

        DrawCmd& drawCmd = gFrameDrawCmds.emplace_back();
        drawCmd.type = DrawCmdType::Draw;
        drawCmd.arg1 = vertexStorage.curBatchSize;
        drawCmd.arg2 = vertexStorage.curBatchStart;
        vertexStorage.endCurrentDrawBatch();
    };

    #if PSYDOOM_GLES_RENDERER
        if (gbUseCpuVertexBuffer) {
            endBatch(gCpuVertexBuffer);
            return;
        }
    #endif

    endBatch(gVertexBuffers_Draw);
}

#if PSYDOOM_GLES_RENDERER
//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: initializes the drawing module to generate vertices into CPU memory, instead of using Vulkan
//------------------------------------------------------------------------------------------------------------------------------------------
void initForGles() noexcept {
    gbUseCpuVertexBuffer = true;
    gCpuVertexBuffer.init((4 * 1024 * 1024) / sizeof(VVertex_Draw));
    gCurDrawPipelineType = (VPipelineType_Main) -1;
    gFrameUniforms.reserve(16);
    gFrameDrawCmds.reserve(4196);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: start of frame logic for the drawing module (main render path)
//------------------------------------------------------------------------------------------------------------------------------------------
void beginFrameGles() noexcept {
    ASSERT(gbUseCpuVertexBuffer);
    ASSERT(gCurDrawPipelineType == (VPipelineType_Main) -1);
    gCpuVertexBuffer.curOffset = 0;
    gCpuVertexBuffer.curBatchStart = 0;
    gCpuVertexBuffer.curBatchSize = 0;
    gCpuVertexBuffer.bInFrame = true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: ends the current batch and gives everything drawn in the frame. Valid until 'endFrameGles'.
//------------------------------------------------------------------------------------------------------------------------------------------
FrameDrawData getFrameDrawDataGles() noexcept {
    ASSERT(gbUseCpuVertexBuffer);
    endCurrentDrawBatch();

    FrameDrawData data = {};
    data.pCmds = gFrameDrawCmds.data();
    data.numCmds = (uint32_t) gFrameDrawCmds.size();
    data.pUniforms = gFrameUniforms.data();
    data.numUniforms = (uint32_t) gFrameUniforms.size();
    data.pVerts = gCpuVertexBuffer.pVerts.get();
    data.numVerts = gCpuVertexBuffer.curOffset;
    return data;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// GLES renderer: end of frame cleanup for the drawing module
//------------------------------------------------------------------------------------------------------------------------------------------
void endFrameGles() noexcept {
    ASSERT(gbUseCpuVertexBuffer);
    gFrameDrawCmds.clear();
    gFrameUniforms.clear();
    gCurDrawPipelineType = (VPipelineType_Main) -1;
    gCpuVertexBuffer.curOffset = 0;
    gCpuVertexBuffer.curBatchStart = 0;
    gCpuVertexBuffer.curBatchSize = 0;
    gCpuVertexBuffer.bInFrame = false;
}
#endif  // #if PSYDOOM_GLES_RENDERER

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a 2D/UI line to the 'draw' subpass
//------------------------------------------------------------------------------------------------------------------------------------------
void addUILine(
    const float x1,
    const float y1,
    const float x2,
    const float y2,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b
) noexcept {
    // Fill in the vertices, starting first with common parameters
    VVertex_Draw* const pVerts = allocDrawVerts(2);

    for (uint32_t i = 0; i < 2; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert = {};
        vert.r = r;
        vert.g = g;
        vert.b = b;
        vert.stmulR = 255;
        vert.stmulG = 255;
        vert.stmulB = 255;
        vert.stmulA = 255;
    }

    // Fill in xy positions
    pVerts[0].x = x1;   pVerts[0].y = y1;
    pVerts[1].x = x2;   pVerts[1].y = y2;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a flat colored triangle to the 'draw' subpass
//------------------------------------------------------------------------------------------------------------------------------------------
void addFlatColoredTriangle(
    const float x1,
    const float y1,
    const float z1,
    const float x2,
    const float y2,
    const float z2,
    const float x3,
    const float y3,
    const float z3,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b
) noexcept {
    VVertex_Draw* const pVerts = allocDrawVerts(3);

    for (uint32_t i = 0; i < 3; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert = {};
        vert.r = r;
        vert.g = g;
        vert.b = b;
        vert.stmulR = 255;
        vert.stmulG = 255;
        vert.stmulB = 255;
        vert.stmulA = 255;
    }

    pVerts[0].x = x1;   pVerts[0].y = y1;   pVerts[0].z = z1;
    pVerts[1].x = x2;   pVerts[1].y = y2;   pVerts[1].z = z2;
    pVerts[2].x = x3;   pVerts[2].y = y3;   pVerts[2].z = z3;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a flat colored quad to the 'draw' subpass
//------------------------------------------------------------------------------------------------------------------------------------------
void addFlatColoredQuad(
    const float x1,
    const float y1,
    const float z1,
    const float x2,
    const float y2,
    const float z2,
    const float x3,
    const float y3,
    const float z3,
    const float x4,
    const float y4,
    const float z4,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b
) noexcept {
    VVertex_Draw* const pVerts = allocDrawVerts(6);

    for (uint32_t i = 0; i < 6; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert = {};
        vert.r = r;
        vert.g = g;
        vert.b = b;
        vert.stmulR = 255;
        vert.stmulG = 255;
        vert.stmulB = 255;
        vert.stmulA = 255;
    }

    pVerts[0].x = x1;   pVerts[0].y = y1;   pVerts[0].z = z1;
    pVerts[1].x = x2;   pVerts[1].y = y2;   pVerts[1].z = z2;
    pVerts[2].x = x3;   pVerts[2].y = y3;   pVerts[2].z = z3;
    pVerts[3].x = x3;   pVerts[3].y = y3;   pVerts[3].z = z3;
    pVerts[4].x = x4;   pVerts[4].y = y4;   pVerts[4].z = z4;
    pVerts[5].x = x1;   pVerts[5].y = y1;   pVerts[5].z = z1;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a UI sprite to the 'draw' subpass
//------------------------------------------------------------------------------------------------------------------------------------------
void addUISprite(
    const float x,
    const float y,
    const float w,
    const float h,
    const float u,
    const float v,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const uint8_t a,
    const uint16_t clutX,
    const uint16_t clutY,
    const uint16_t texPageX,
    const uint16_t texPageY,
    const uint16_t texWinW,
    const uint16_t texWinH
) noexcept {
    // Fill in the vertices, starting first with common parameters
    VVertex_Draw* const pVerts = allocDrawVerts(6);

    for (uint32_t i = 0; i < 6; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert.z = {};                    // Unused for UI shaders
        vert.r = r;
        vert.g = g;
        vert.b = b;
        vert.lightDimModeFlags = {};    // Unused for UI shaders
        vert.texWinX = texPageX;
        vert.texWinY = texPageY;
        vert.texWinW = texWinW;
        vert.texWinH = texWinH;
        vert.clutX = clutX;
        vert.clutY = clutY;
        vert.stmulR = 128;              // Fully white (128 = 100% strength)
        vert.stmulG = 128;
        vert.stmulB = 128;
        vert.stmulA = a;
    }

    // Fill in the xy positions
    const float xl = x;
    const float xr = x + w;
    const float yt = y;
    const float yb = y + h;

    const float ul = u;
    const float ur = u + w;
    const float vt = v;
    const float vb = v + h;

    pVerts[0].x = xl;   pVerts[0].y = yt;
    pVerts[1].x = xr;   pVerts[1].y = yt;
    pVerts[2].x = xr;   pVerts[2].y = yb;
    pVerts[3].x = xr;   pVerts[3].y = yb;
    pVerts[4].x = xl;   pVerts[4].y = yb;
    pVerts[5].x = xl;   pVerts[5].y = yt;

    pVerts[0].u = ul;   pVerts[0].v = vt;
    pVerts[1].u = ur;   pVerts[1].v = vt;
    pVerts[2].u = ur;   pVerts[2].v = vb;
    pVerts[3].u = ur;   pVerts[3].v = vb;
    pVerts[4].u = ul;   pVerts[4].v = vb;
    pVerts[5].u = ul;   pVerts[5].v = vt;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a triangle for the game's 3D view/world to the 'draw' subpass.
// 
// Notes:
//  (1) The texture format is assumed to be 8 bits per pixel always.
//  (2) All texture coordinates and texture sizes are in terms of 8-bit pixels (not VRAM 16-bit pixels).
//  (3) The alpha component is only used if alpha blending is being used.
//------------------------------------------------------------------------------------------------------------------------------------------
void addWorldTriangle(
    const float x1,
    const float y1,
    const float z1,
    const float u1,
    const float v1,
    const float x2,
    const float y2,
    const float z2,
    const float u2,
    const float v2,
    const float x3,
    const float y3,
    const float z3,
    const float u3,
    const float v3,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const uint16_t clutX,
    const uint16_t clutY,
    const uint16_t texWinX,
    const uint16_t texWinY,
    const uint16_t texWinW,
    const uint16_t texWinH,
    const uint8_t lightDimModeFlags,
    const uint8_t stMulR,
    const uint8_t stMulG,
    const uint8_t stMulB,
    const uint8_t stMulA
) noexcept {
    // Fill in the vertices, starting first with common parameters
    VVertex_Draw* const pVerts = allocDrawVerts(3);

    for (uint32_t i = 0; i < 3; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert.r = r;
        vert.g = g;
        vert.b = b;
        vert.texWinX = texWinX;
        vert.texWinY = texWinY;
        vert.texWinW = texWinW;
        vert.texWinH = texWinH;
        vert.clutX = clutX;
        vert.clutY = clutY;
        vert.stmulR = stMulR;
        vert.stmulG = stMulG;
        vert.stmulB = stMulB;
        vert.stmulA = stMulA;
        vert.lightDimModeFlags = lightDimModeFlags;
    }

    // Fill in verts xy and uv positions
    pVerts[0].x = x1;   pVerts[0].y = y1;   pVerts[0].z = z1;
    pVerts[1].x = x2;   pVerts[1].y = y2;   pVerts[1].z = z2;
    pVerts[2].x = x3;   pVerts[2].y = y3;   pVerts[2].z = z3;

    pVerts[0].u = u1;   pVerts[0].v = v1;
    pVerts[1].u = u2;   pVerts[1].v = v2;
    pVerts[2].u = u3;   pVerts[2].v = v3;
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a quadrilateral for the game's 3D view/world to the 'draw' subpass.
// 
// Notes:
//  (1) The texture format is assumed to be 8 bits per pixel always.
//  (2) All texture coordinates and texture sizes are in terms of 8-bit pixels (not VRAM 16-bit pixels).
//  (3) The alpha component is only used if alpha blending is being used.
//------------------------------------------------------------------------------------------------------------------------------------------
void addWorldQuad(
    const AddWorldQuadVert& v1,
    const AddWorldQuadVert& v2,
    const AddWorldQuadVert& v3,
    const AddWorldQuadVert& v4,
    const uint16_t clutX,
    const uint16_t clutY,
    const uint16_t texWinX,
    const uint16_t texWinY,
    const uint16_t texWinW,
    const uint16_t texWinH,
    const uint8_t lightDimModeFlags,
    const uint8_t stMulR,
    const uint8_t stMulG,
    const uint8_t stMulB,
    const uint8_t stMulA
) noexcept {
    // Fill in the vertices, starting first with the parameters that are the same for all vertices
    VVertex_Draw* const pVerts = allocDrawVerts(6);

    for (uint32_t i = 0; i < 6; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert.texWinX = texWinX;
        vert.texWinY = texWinY;
        vert.texWinW = texWinW;
        vert.texWinH = texWinH;
        vert.clutX = clutX;
        vert.clutY = clutY;
        vert.stmulR = stMulR;
        vert.stmulG = stMulG;
        vert.stmulB = stMulB;
        vert.stmulA = stMulA;
        vert.lightDimModeFlags = lightDimModeFlags;
    }

    // Fill in the unique vertex attributes
    constexpr auto assignVertexUniqueAttribs = [](VVertex_Draw& dst, const AddWorldQuadVert& src) noexcept {
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.u = src.u;
        dst.v = src.v;
        dst.r = src.r;
        dst.g = src.g;
        dst.b = src.b;
    };

    assignVertexUniqueAttribs(pVerts[0], v1);
    assignVertexUniqueAttribs(pVerts[1], v2);
    assignVertexUniqueAttribs(pVerts[2], v3);
    assignVertexUniqueAttribs(pVerts[3], v3);
    assignVertexUniqueAttribs(pVerts[4], v4);
    assignVertexUniqueAttribs(pVerts[5], v1);
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a vertical quad for the sky to the 'draw' subpass.
// The y coordinate where the sky starts and the 2 endpoints are specified only, along with whether it is an upper or lower sky wall.
// The sky wall is stretched past the top or bottom of the screen.
//------------------------------------------------------------------------------------------------------------------------------------------
void addWorldInfiniteSkyWall(
    const float x1,
    const float z1,
    const float x2,
    const float z2,
    const float y,
    const bool bIsUpperSkyWall,
    const float skyUOffset,
    const uint16_t clutX,
    const uint16_t clutY,
    const uint16_t texWinX,
    const uint16_t texWinY,
    const uint16_t texWinW,
    const uint16_t texWinH
) noexcept {
    // Fill in the vertices, starting first with common parameters.
    // Note: we store the sky U offset based on player rotation in the U coordinate.
    VVertex_Draw* const pVerts = allocDrawVerts(6);

    for (uint32_t i = 0; i < 6; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert.y = y;
        vert.r = 128;
        vert.g = 128;
        vert.b = 128;
        vert.lightDimModeFlags = {};    // Unused for sky rendering
        vert.u = skyUOffset;
        vert.texWinX = texWinX;
        vert.texWinY = texWinY;
        vert.texWinW = texWinW;
        vert.texWinH = texWinH;
        vert.clutX = clutX;
        vert.clutY = clutY;
        vert.stmulR = 128;
        vert.stmulG = 128;
        vert.stmulB = 128;
        vert.stmulA = 128;
    }

    // Note: the 'v' coordinate is used to determine whether the vertex is for the bottom or the top of the sky wall.
    // If the 'v' coord is '-1' or '+1' then it means the vertex should be stretched past the top (+1) or bottom (-1) of the screen.
    pVerts[0].x = x1;   pVerts[0].z = z1;
    pVerts[1].x = x2;   pVerts[1].z = z2;
    pVerts[2].x = x2;   pVerts[2].z = z2;
    pVerts[3].x = x2;   pVerts[3].z = z2;
    pVerts[4].x = x1;   pVerts[4].z = z1;
    pVerts[5].x = x1;   pVerts[5].z = z1;

    // Make the skywall be a certain fixed height to begin with.
    // This should be enough for most situations but the vertex shader will also stretch the wall past the ends of the screen too.
    constexpr float EXTRUDE_AMT = 1024.0f;

    if (bIsUpperSkyWall) {
        pVerts[0].y = y;                pVerts[0].v = 0.0f;
        pVerts[1].y = y + EXTRUDE_AMT;  pVerts[1].v = 1.0f;
        pVerts[2].y = y;                pVerts[2].v = 0.0f;
        pVerts[3].y = y + EXTRUDE_AMT;  pVerts[3].v = 1.0f;
        pVerts[4].y = y;                pVerts[4].v = 0.0f;
        pVerts[5].y = y + EXTRUDE_AMT;  pVerts[5].v = 1.0f;
    } else {
        pVerts[0].y = y - EXTRUDE_AMT;  pVerts[0].v = -1.0f;
        pVerts[1].y = y;                pVerts[1].v =  0.0f;
        pVerts[2].y = y - EXTRUDE_AMT;  pVerts[2].v = -1.0f;
        pVerts[3].y = y;                pVerts[3].v =  0.0f;
        pVerts[4].y = y - EXTRUDE_AMT;  pVerts[4].v = -1.0f;
        pVerts[5].y = y;                pVerts[5].v =  0.0f;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------
// Add a quad for the sky to the 'draw' subpass. Unlike the infinite sky wall, this quad is not stretched in any way
//------------------------------------------------------------------------------------------------------------------------------------------
void addWorldSkyQuad(
    const float x1,
    const float y1,
    const float z1,
    const float x2,
    const float y2,
    const float z2,
    const float x3,
    const float y3,
    const float z3,
    const float x4,
    const float y4,
    const float z4,
    const float skyUOffset,
    const uint16_t clutX,
    const uint16_t clutY,
    const uint16_t texWinX,
    const uint16_t texWinY,
    const uint16_t texWinW,
    const uint16_t texWinH
) noexcept {
    // Fill in the vertices, starting first with common parameters.
    // Note: we store the sky U offset based on player rotation in the U coordinate.
    VVertex_Draw* const pVerts = allocDrawVerts(6);

    for (uint32_t i = 0; i < 6; ++i) {
        VVertex_Draw& vert = pVerts[i];
        vert.r = 128;
        vert.g = 128;
        vert.b = 128;
        vert.lightDimModeFlags = {};    // Unused for sky rendering
        vert.u = skyUOffset;
        vert.v = 0.0f;
        vert.texWinX = texWinX;
        vert.texWinY = texWinY;
        vert.texWinW = texWinW;
        vert.texWinH = texWinH;
        vert.clutX = clutX;
        vert.clutY = clutY;
        vert.stmulR = 128;
        vert.stmulG = 128;
        vert.stmulB = 128;
        vert.stmulA = 128;
    }

    // Fill the rest of the coords
    pVerts[0].x = x1;   pVerts[0].y = y1;   pVerts[0].z = z1;
    pVerts[1].x = x2;   pVerts[1].y = y2;   pVerts[1].z = z2;
    pVerts[2].x = x3;   pVerts[2].y = y3;   pVerts[2].z = z3;
    pVerts[3].x = x3;   pVerts[3].y = y3;   pVerts[3].z = z3;
    pVerts[4].x = x4;   pVerts[4].y = y4;   pVerts[4].z = z4;
    pVerts[5].x = x1;   pVerts[5].y = y1;   pVerts[5].z = z1;
}

END_NAMESPACE(VDrawing)

#endif  // #if PSYDOOM_VULKAN_RENDERER
