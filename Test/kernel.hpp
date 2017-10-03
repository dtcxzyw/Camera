#pragma once
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <Base/Model.hpp> 

using VI = StaticMesh::Vertex;
using OI =EmptyArg;
 
struct FrameBufferGPU final {
    BuiltinRenderTargetGPU<RGBA> color;
    BuiltinRenderTargetGPU<float> depth;
    uvec2 mSize;
    CUDA uvec2 size() const {
        return mSize;
    }
};

struct FrameBufferCPU final {
    template<typename T>
    using UIT = std::unique_ptr<BuiltinRenderTarget<T>>;
    UIT<RGBA> colorBuffer;
    UIT<float> depthBuffer;
    uvec2 size;
    DataViewer<FrameBufferGPU> dataGPU;
    void resize(size_t width, size_t height) {
        if (size.x == width && size.y == height)return;
        colorBuffer = std::make_unique<BuiltinRenderTarget<RGBA>>(width, height);
        depthBuffer = std::make_unique<BuiltinRenderTarget<float>>(width, height);
        FrameBufferGPU data;
        data.color = colorBuffer->toTarget();
        data.depth = depthBuffer->toTarget();
        data.mSize = size= { width,height };
        dataGPU = share(std::vector<FrameBufferGPU>{ data });
    }
};
 
struct Uniform final {
    ALIGN mat4 mat;
};

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo, DataViewer<Uniform> uniform
    ,FrameBufferCPU& fbo,Pipeline& pipeline);

