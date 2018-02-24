#pragma once
#include <Interaction/D3D11.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/Builtin.hpp>
#include <PBR/PhotorealisticRendering.hpp>
#include <PBR/BRDF.hpp>

struct FrameBufferGPU final {
    BuiltinRenderTargetGPU<RGBA> color;
    uvec2 fsize;
    CUDAINLINE uvec2 size() const {
        return fsize;
    }
};

struct FrameBufferCPU final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<BuiltinRenderTarget<RGBA>> colorRT;
    D3D11Image image;
    uvec2 size;
    FrameBufferGPU data;

    void resize(const uvec2 nsiz) {
        if (size == nsiz)return;
        size = nsiz;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(size,cudaArraySurfaceLoadStore);
        colorRT = std::make_unique<BuiltinRenderTarget<RGBA>>(*colorBuffer);
        image.resize(size);
        data.color = colorRT->toTarget();
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

struct PostUniform final {
    ALIGN FrameBufferGPU in;
    ALIGN float* lum;
    ALIGN std::pair<float, unsigned int>* sum;

    PostUniform(const FrameBufferGPU buf, float* clum, std::pair<float, unsigned int>* cnt)
        : in(buf), lum(clum), sum(cnt) {}
};

