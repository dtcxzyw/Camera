#pragma once
#include <Core/CommandBuffer.hpp>
#include <Core/DataSet.hpp>
#include <Core/Builtin.hpp>
#include <Core/Buffer2D.hpp>
#include <IO/Model.hpp>
#include <BxDF/BRDFOld.hpp>
#include <Camera/PinholeCamera.hpp>
#include <Rasterizer/TriangleRasterizer.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Light/DeltaPositionLight.hpp>
#include <Rasterizer/SphereRasterizer.hpp>

using VI = VertexDesc;

enum OutInfo {
    Pos,
    Nor,
    Tangent,
    TexCoord
};

using OI = Args<VAR(Pos, Point), VAR(Nor, Normal), VAR(Tangent, Normal)>;

struct FrameBufferRef final {
    BuiltinRenderTargetRef<RGBA> color;
    Buffer2DRef<uint32_t> depth;
    uvec2 fsize;
    FrameBufferRef() = default;

    FrameBufferRef(const FrameBufferRef& rhs, const Buffer2DRef<uint32_t> buf) {
        color = rhs.color;
        depth = buf;
        fsize = rhs.fsize;
    }

    DEVICEINLINE uvec2 size() const {
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
        data.color = colorRT->toRef();
        data.fsize = size;
    }

    Span<FrameBufferRef> getData(CommandBuffer& buffer,
        Buffer2D<uint32_t>& depth) const {
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
    ALIGN Transform cameraTransform;
    ALIGN Transform skyTransform;
    ALIGN Transform modelTransform;
    ALIGN Transform invCameraTransform;
    ALIGN Point cp;
    ALIGN DisneyBRDFArg arg;
    ALIGN BuiltinSamplerRef<RGBA> sampler;
    ALIGN PointLight light;
};

struct PostUniform final {
    ALIGN FrameBufferRef in;
    ALIGN float* lum;
    ALIGN std::pair<float, uint32_t>* sum;

    PostUniform(const FrameBufferRef buf, float* clum, std::pair<float, uint32_t>* cnt)
        : in(buf), lum(clum), sum(cnt) {}
};

struct RenderingContext final {
    VersionCounter vertCounter;
    VertexCache<OI, VersionComparer> cache;
    TriangleRenderingContext<OI, VersionComparer> triContext;

    CacheRef<MemorySpan<VertexInfo<OI>>, VersionComparer> get() {
        return {cache, vertCounter.get()};
    }
};

void kernel(const StaticMeshData& model, RenderingContext& mc,
    const StaticMeshData& skybox, RenderingContext& sc,
    const MemorySpan<SphereDesc>& spheres,
    const Span<Uniform>& uniform, FrameBuffer& fbo, float* lum,
    PinholeCamera::RasterPosConverter converter, CommandBuffer& buffer);
