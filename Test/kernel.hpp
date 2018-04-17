#pragma once
#include <Core/DispatchSystem.hpp>
#include <Core/DataSet.hpp>
#include <Core/Builtin.hpp>
#include <Rasterizer/Buffer2D.hpp>
#include <IO/Model.hpp>
#include <PBR/BRDF.hpp>
#include <Camera/PinholeCamera.hpp>
#include <Rasterizer/TriangleRasterizer.hpp>
#include <Spectrum/RGBSpectrum.hpp>
#include <Light/DeltaPositionLight.hpp>

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
    FrameBufferRef() = default;
    FrameBufferRef(const FrameBufferRef& rhs, const Buffer2DRef<unsigned int> buf) {
        color = rhs.color;
        depth = buf;
        fsize = rhs.fsize;
    }
    CUDAINLINE uvec2 size() const {
        return fsize;
    }
};

struct FrameBuffer final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<BuiltinArray<RGBA8>> postBuffer;
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
        data.color = colorRT->toTarget();
        data.fsize = size;
    }

    Span<FrameBufferRef> getData(CommandBuffer& buffer,
        Buffer2D<unsigned int>& depth) const {
        auto dataRef = buffer.allocConstant<FrameBufferRef>();
        auto&& manager = buffer.getResourceManager();
        buffer.memcpy(dataRef, [this,&manager, depthRef= depth.toBuffer()](auto call) {
            auto buf = data;
            buf.depth = depthRef.get(manager);
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
    ALIGN DisneyBRDFArg<RGBSpectrum> arg;
    ALIGN BuiltinSamplerRef<RGBA> sampler;
    ALIGN PointLight<RGBSpectrum> light;
};

struct PostUniform final {
    ALIGN FrameBufferRef in;
    ALIGN float* lum;
    ALIGN std::pair<float, unsigned int>* sum;

    PostUniform(const FrameBufferRef buf, float* clum, std::pair<float, unsigned int>* cnt)
        : in(buf), lum(clum), sum(cnt) {}
};

struct RenderingContext final {
    VersionCounter vertCounter;
    VertexCache<OI, VersionComparer> cache;
    TriangleRenderingContext<OI, VersionComparer> triContext;
    CacheRef<MemorySpan<VertexInfo<OI>>, VersionComparer> get() {
        return { cache,vertCounter.get() };
    }
};

void kernel(const StaticMesh& model, RenderingContext& mc,
            const StaticMesh& skybox, RenderingContext& sc,
            const MemorySpan<vec4>& spheres,
            const Span<Uniform>& uniform, FrameBuffer& fbo, float* lum,
            PinholeCamera::RasterPosConverter converter, CommandBuffer& buffer);
