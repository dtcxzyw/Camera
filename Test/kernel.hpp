#pragma once
#include <Core/DispatchSystem.hpp>
#include <Core/DataSet.hpp>
#include <Core/Builtin.hpp>
#include <Rasterizer/Buffer2D.hpp>
#include <IO/Model.hpp>
#include <BxDF/BRDFOld.hpp>
#include <Camera/PinholeCamera.hpp>
#include <Rasterizer/TriangleRasterizer.hpp>
#include <Spectrum/RGBSpectrum.hpp>
#include <Light/DeltaPositionLight.hpp>
#include <Rasterizer/SphereRasterizer.hpp>

using VI = VertexDesc;

enum OutInfo {
    Pos,
    Nor,
    Tangent,
    TexCoord
};

using OI = Args<VAR(Pos, Point), VAR(Nor, Vector), VAR(Tangent, Vector)>;

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
    ALIGN std::pair<float, unsigned int>* sum;

    PostUniform(const FrameBufferRef buf, float* clum, std::pair<float, unsigned int>* cnt)
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
