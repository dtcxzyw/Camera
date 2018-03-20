#pragma once
#include <Interaction/D3D11.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <ScanLineRenderer/Buffer2D.hpp>
#include <IO/Model.hpp>
#include <PBR/BRDF.hpp>
#include <PBR/PhotorealisticRendering.hpp>
#include <ScanLineRenderer/RenderingCache.hpp>
#include <ScanLineRenderer/TriangleRasterizer.hpp>

using VI = StaticMesh::Vertex;

enum OutInfo {
    Pos,
    Normal,
    Tangent,
    TexCoord
};

using OI = Args<VAR(Pos, vec3), VAR(Normal, vec3),VAR(Tangent, vec3)>;

struct FrameBufferRef final {
    BuiltinRenderTargetRef<RGBA> color;
    Buffer2DRef<unsigned int> depth;
    uvec2 fsize;
    CUDAINLINE uvec2 size() const {
        return fsize;
    }
};

struct FrameBuffer final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<BuiltinArray<RGBA8>> postBuffer;
    std::unique_ptr<Buffer2D<unsigned int>> depthBuffer;
    std::unique_ptr<BuiltinRenderTarget<RGBA>> colorRT;
    std::unique_ptr<BuiltinRenderTarget<RGBA8>> postRT;
    uvec2 size;
    FrameBufferRef data;

    void resize(const uvec2 nsiz) {
        if (size == nsiz)return;
        size = nsiz;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(size,cudaArraySurfaceLoadStore);
        colorRT = std::make_unique<BuiltinRenderTarget<RGBA>>(*colorBuffer);
        postBuffer = std::make_unique<BuiltinArray<RGBA8>>(size, cudaArraySurfaceLoadStore);
        postRT = std::make_unique<BuiltinRenderTarget<RGBA8>>(*postBuffer);
        depthBuffer = std::make_unique<Buffer2D<unsigned int>>(size);
        data.color = colorRT->toTarget();
        data.depth = depthBuffer->toBuffer();
        data.fsize = size;
    }

    MemoryRef<FrameBufferRef> getData(CommandBuffer& buffer) const {
        auto dataRef = buffer.allocConstant<FrameBufferRef>();
        buffer.memcpy(dataRef, [buf=data](auto call) {
            call(&buf);
        });
        return dataRef;
    }
};

struct Uniform final {
    ALIGN vec2 mul;
    ALIGN mat4 M;
    ALIGN mat4 Msky;
    ALIGN mat4 V;
    ALIGN mat4 invV;
    ALIGN mat3 normalInvV;
    ALIGN mat3 normalMat;
    ALIGN vec3 cp;
    ALIGN vec3 lp;
    ALIGN vec3 lc;
    ALIGN float r2;
    ALIGN DisneyBRDFArg arg;
    ALIGN BuiltinSamplerRef<RGBA> sampler;
    ALIGN RC8::BlockRef cache;
};

struct PostUniform final {
    ALIGN FrameBufferRef in;
    ALIGN float* lum;
    ALIGN std::pair<float, unsigned int>* sum;

    PostUniform(const FrameBufferRef buf, float* clum, std::pair<float, unsigned int>* cnt)
        : in(buf), lum(clum), sum(cnt) {}
};

void kernel(const StaticMesh& model, TriangleRenderingHistory& mh,
            const StaticMesh& skybox, TriangleRenderingHistory& sh,
            const DataViewer<vec4>& spheres,
            const MemoryRef<Uniform>& uniform, FrameBuffer& fbo, float* lum,
            Camera::RasterPosConverter converter, CommandBuffer& buffer);
