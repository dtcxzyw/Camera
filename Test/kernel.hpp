#pragma once
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <ScanLineRenderer/DepthBuffer.hpp>
#include <IO/Model.hpp> 
#include <PostProcess/ToneMapping.hpp>
#include <Base/Constant.hpp>

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
    Constant<FrameBufferGPU> dataGPU;
    FrameBufferGPU data;
    void resize(size_t width, size_t height,Stream& stream) {
        if (size.x == width && size.y == height)return;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(width, height);
        colorRT = std::make_unique <BuiltinRenderTarget<RGBA>>(*colorBuffer);
        depthBuffer = std::make_unique<DepthBuffer<unsigned int>>(uvec2(width, height));
        data.color = colorRT->toTarget();
        data.depth = depthBuffer->toBuffer();
        data.mSize = size = { width,height };
        dataGPU.set(data,stream);
    }
};
 
struct Uniform final {
    ALIGN mat4 VP;
    ALIGN mat4 M;
    ALIGN mat3 invM;
    ALIGN vec3 cp;
    ALIGN vec3 dir;
    ALIGN vec3 lc;
    ALIGN vec3 albedo;
    ALIGN float roughness;
    ALIGN float metallic;
    ALIGN float ao;
};

struct PostUniform final {
    ALIGN FrameBufferGPU in;
    ALIGN float lum;
    ALIGN std::pair<float, unsigned int>* sum;
};

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo, const Uniform* uniform
    ,FrameBufferCPU& fbo,const PostUniform* post,
    BuiltinRenderTargetGPU<RGBA> dest, Stream& stream);


