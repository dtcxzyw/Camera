#pragma once
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <IO/Model.hpp> 

using VI = StaticMesh::Vertex;
using OI = EmptyArg;
 
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
    using UBR = std::unique_ptr<BuiltinArray<T>>;
    template<typename T>
    using URT = std::unique_ptr<BuiltinRenderTarget<T>>;
    UBR<RGBA> colorBuffer;
    UBR<float> depthBuffer;
    URT<RGBA> colorRT;
    URT<float> depthRT;
    uvec2 size;
    DataViewer<FrameBufferGPU> dataGPU;
    void resize(size_t width, size_t height) {
        if (size.x == width && size.y == height)return;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(width, height);
        colorRT = std::make_unique <BuiltinRenderTarget<RGBA>>(*colorBuffer);
        depthBuffer = std::make_unique<BuiltinArray<float>>(width, height);
        depthRT = std::make_unique <BuiltinRenderTarget<float>>(*depthBuffer);
        FrameBufferGPU data;
        data.color = colorRT->toTarget();
        data.depth = depthRT->toTarget();
        data.mSize = size= { width,height };
        dataGPU = share(std::vector<FrameBufferGPU>{ data });
    }
};
 
struct Uniform final {
    ALIGN mat4 mat;
};

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo, DataViewer<Uniform> uniform
    ,FrameBufferCPU& fbo,Pipeline& pipeline);

