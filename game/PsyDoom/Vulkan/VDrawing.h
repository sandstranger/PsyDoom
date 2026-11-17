#pragma once

#if PSYDOOM_VULKAN_RENDERER

#include "Macros.h"
#include "Matrix4.h"

#include <cstdint>

namespace vgl {
    class BaseRenderPass;
    class BaseTexture;
    class CmdBufferRecorder;
    class Framebuffer;
    class LogicalDevice;
}

namespace Gpu {
    enum class BlendMode : uint8_t;
    enum class TexFmt : uint8_t;
}

enum class VPipelineType_Main : uint8_t;
struct VShaderUniforms_Draw;
struct VVertex_Draw;

#if PSYDOOM_GLES_RENDERER
    struct VInstance_Draw;
#endif

BEGIN_NAMESPACE(VDrawing)

// Attributes which are specified uniquely per vertex for 'addWorldQuad'
struct AddWorldQuadVert {
    float       x, y, z;
    float       u, v;
    uint8_t     r, g, b;
};

//------------------------------------------------------------------------------------------------------------------------------------------
// Enum representing a type of draw command. We store submitted draw commands in a list and record the Vulkan command buffer at the end
// of the frame, once the vertex buffers are finalized and have known sizes. The GLES renderer replays the same list at the end of the frame.
//------------------------------------------------------------------------------------------------------------------------------------------
enum class DrawCmdType : uint32_t {
    SetPipeline,        // Set the graphics pipeline to use: 1st arg is pipeline type, 2nd arg unused
    SetUniforms,        // Set the uniforms to use: 1st arg is index in the uniforms list
    Draw                // A command to draw primitives: 1st arg is vertex count, 2nd arg is vertex offset
};

//------------------------------------------------------------------------------------------------------------------------------------------
// Stores a single drawing command: the meaning of the arguments depends on the command type
//------------------------------------------------------------------------------------------------------------------------------------------
struct DrawCmd {
    DrawCmdType     type;
    uint32_t        arg1;
    uint32_t        arg2;
};

#if PSYDOOM_GLES_RENDERER
    //--------------------------------------------------------------------------------------------------------------------------------------
    // GLES renderer: everything drawn in the frame, for it to upload and replay at the end of the frame.
    // With the GLES renderer vertices are generated into CPU memory and uploaded once per frame (no buffers mapped while drawing).
    // With instancing (OpenGL ES 3.0) each primitive is an instance instead ('VInstance_Draw'): the vertex counts and offsets of the
    // draw commands are then instance counts and offsets.
    //--------------------------------------------------------------------------------------------------------------------------------------
    struct FrameDrawData {
        const DrawCmd*                  pCmds;
        uint32_t                        numCmds;
        const VShaderUniforms_Draw*     pUniforms;
        uint32_t                        numUniforms;
        const VVertex_Draw*             pVerts;
        uint32_t                        numVerts;
        const VInstance_Draw*           pInstances;
        uint32_t                        numInstances;
        bool                            bInstanced;
    };

    void initForGles(const bool bUseInstancing) noexcept;
    void beginFrameGles() noexcept;
    FrameDrawData getFrameDrawDataGles() noexcept;
    void endFrameGles() noexcept;
#endif

void init(vgl::LogicalDevice& device, vgl::BaseTexture& vramTex) noexcept;
void shutdown() noexcept;
void beginFrame(const uint32_t ringbufferIdx) noexcept;
void endFrame(vgl::CmdBufferRecorder& cmdRec, const bool bRenderThisFrame) noexcept;
bool hasIssuedDrawCmds() noexcept;
void setDrawPipeline(const VPipelineType_Main type) noexcept;
void setDrawUniforms(const VShaderUniforms_Draw& uniforms) noexcept;
Matrix4f computeTransformMatrixForUI(const bool bAllowWidescreen) noexcept;
Matrix4f computeTransformMatrixFor3D(const float viewX, const float viewY, const float viewZ, const float viewAngle) noexcept;
void endCurrentDrawBatch() noexcept;

void addUILine(
    const float x1,
    const float y1,
    const float x2,
    const float y2,
    const uint8_t r,
    const uint8_t g,
    const uint8_t b
) noexcept;

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
) noexcept;

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
) noexcept;

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
) noexcept;

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
) noexcept;

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
) noexcept;

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
) noexcept;

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
) noexcept;

END_NAMESPACE(VDrawing)

#endif  // #if PSYDOOM_VULKAN_RENDERER
