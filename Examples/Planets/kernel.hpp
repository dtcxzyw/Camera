#pragma once
#include <Interaction/D3D11.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/Builtin.hpp>
#include <ScanLineRenderer/DepthBuffer.hpp>
#include <PBR/PhotorealisticRendering.hpp>

struct FrameBufferGPU final {
    BuiltinRenderTargetGPU<RGBA> color;
    DepthBufferGPU<unsigned int> depth;
    uvec2 fsize;
    CUDAINLINE uvec2 size() const {
        return fsize;
    }
};

struct FrameBufferCPU final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<BuiltinArray<RGBA8>> postBuffer;
    std::unique_ptr<DepthBuffer<unsigned int>> depthBuffer;
    std::unique_ptr<BuiltinRenderTarget<RGBA>> colorRT;
    std::unique_ptr<BuiltinRenderTarget<RGBA8>> postRT;
    uvec2 size;
    FrameBufferGPU data;

    void resize(const uvec2 nsiz) {
        if (size == nsiz)return;
        size = nsiz;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(size,cudaArraySurfaceLoadStore);
        colorRT = std::make_unique<BuiltinRenderTarget<RGBA>>(*colorBuffer);
        postBuffer = std::make_unique<BuiltinArray<RGBA8>>(size, cudaArraySurfaceLoadStore);
        postRT = std::make_unique<BuiltinRenderTarget<RGBA8>>(*postBuffer);
        depthBuffer = std::make_unique<DepthBuffer<unsigned int>>(size);
        data.color = colorRT->toTarget();
        data.depth = depthBuffer->toBuffer();
        data.fsize = size;
    }

    MemoryRef<FrameBufferGPU> getData(CommandBuffer& buffer) const {
        auto dataGPU = buffer.allocConstant<FrameBufferGPU>();
        buffer.memcpy(dataGPU, [buf=data](auto call) {
            call(&buf);
        });
        return dataGPU;
    }
};

struct Uniform final {
    ALIGN mat4 V;
};

void kernel(const MemoryRef<vec4>& spheres,
            const MemoryRef<Uniform>& uniform, FrameBufferCPU& fbo,
            Camera::RasterPosConverter converter, CommandBuffer& buffer);
