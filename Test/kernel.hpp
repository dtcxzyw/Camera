#pragma once
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <ScanLineRenderer/DepthBuffer.hpp>
#include <IO/Model.hpp> 

using VI = StaticMesh::Vertex;
enum OutInfo {
    pos,normal
};
using OI = Args<Var(pos,vec3),Var(normal,vec3)>;
 
struct FrameBufferGPU final {
    BuiltinRenderTargetGPU<RGBA> color;
    DepthBufferGPU<unsigned int> depth;
    uvec2 mSize;
    CUDA uvec2 size() const {
        return mSize;
    }
};

struct FrameBufferCPU final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<DepthBuffer<unsigned int>> depthBuffer;
    std::unique_ptr<BuiltinRenderTarget<RGBA>> colorRT;
    uvec2 size;
    DataViewer<FrameBufferGPU> dataGPU;
    void resize(size_t width, size_t height) {
        if (size.x == width && size.y == height)return;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(width, height);
        colorRT = std::make_unique <BuiltinRenderTarget<RGBA>>(*colorBuffer);
        depthBuffer = std::make_unique<DepthBuffer<unsigned int>>(uvec2(width, height));
        dataGPU = allocBuffer<FrameBufferGPU>(1);
        auto& data=dataGPU[0];
        data.color = colorRT->toTarget();
        data.depth = depthBuffer->toBuffer();
        data.mSize = size = { width,height };
    }
};
 
struct Uniform final {
    ALIGN mat4 VP;
    ALIGN mat4 M;
    ALIGN mat3 invM;
    ALIGN vec3 cp;
    ALIGN vec3 dir;
    ALIGN vec3 lc;
    ALIGN vec3 color;
    ALIGN float roughness;
    ALIGN vec3 f0;
    ALIGN float albedo;
    ALIGN float metallic;
};

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo, DataViewer<Uniform> uniform
    ,FrameBufferCPU& fbo, BuiltinRenderTargetGPU<RGBA> dest, Pipeline& pipeline);

