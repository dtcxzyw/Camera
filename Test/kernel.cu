#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <sm_20_intrinsics.h>

CUDA void VS(VI in, Uniform uniform, vec4& pos, OI& out) {
    pos = uniform.mat* vec4(in.pos, 1.0f);
}

CUDA void drawPoint(ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    uv.y = fbo.mSize.y - uv.y;
    if (fbo.depth.get(uv) > z) {
        constexpr auto v = 0.5f;
        auto g =1.5f- (z - v) / (1.0f - v);
        vec4 color = { g,g,g,1.0f };
        for (int i = 0; i < 4; ++i) {
            auto fz = fbo.depth.get(uv);
            if (fz > z | (fz==z & fbo.color.get(uv) != color)) {
                fbo.depth.set(uv, z);
                __threadfence_system();
                fbo.color.set(uv, color);
                __threadfence_system();
            }
        }
    }
}

CUDA void post(ivec2 NDC, FrameBufferGPU in, BuiltinRenderTargetGPU<RGBA> out) {
    constexpr int rad = 5;
    constexpr auto base = 5.0f, sub =base*(rad*2+1)*(rad*2+1)-1.0f;
    float w = 0.0f;
    for (int i =  - rad; i <= rad; ++i)
        for (int j = - rad; j <=  rad; ++j)
            w +=in.depth.get({NDC.x+i,NDC.y+j });
    w = w*base - sub*in.depth.get(NDC);
    auto c = in.color.get(NDC);
    out.set(NDC, c*w);
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo,DataViewer<Uniform> uniform,
    FrameBufferCPU& fbo, BuiltinRenderTargetGPU<RGBA> dest,Pipeline& pipeline) {
    fbo.colorRT->clear(pipeline,vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthRT->clear(pipeline,1.0f);
    renderTriangles<VI, OI, Uniform, FrameBufferGPU, VS, drawPoint>
        (pipeline,vbo, ibo, uniform, fbo.dataGPU,fbo.size);
    renderFullScreen<FrameBufferGPU, decltype(dest), post>(pipeline,fbo.dataGPU,dest,fbo.size);
}

