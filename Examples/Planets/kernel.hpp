#pragma once
#include <Core/DispatchSystem.hpp>
#include <Core/Builtin.hpp>
#include <Camera/PinholeCamera.hpp>

struct FrameBufferRef final {
    BuiltinRenderTargetRef<RGBA> color;
    uvec2 fsize;
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

    Span<FrameBufferRef> getData(CommandBuffer& buffer) const {
        auto dataRef = buffer.allocConstant<FrameBufferRef>();
        buffer.memcpy(dataRef, [buf=data](auto call) {
            call(&buf);
        });
        return dataRef;
    }
};

struct Uniform final {
    ALIGN mat4 V;
};

void kernel(const Span<vec4>& spheres,
            const Span<Uniform>& uniform, FrameBuffer& fbo,
            PinholeCamera::RasterPosConverter converter, CommandBuffer& buffer);
