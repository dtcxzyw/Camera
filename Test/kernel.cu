#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"

CUDA void VS(VI in, Uniform uniform, vec4& pos, OI& out) {
    pos = uniform.mat* vec4(in.pos, 1.0f);
}

CUDA void setPoint(ivec2 uv, unsigned int z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z);
}

CUDA void drawPoint(ivec2 uv, unsigned int z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) == z) {
        vec4 color = { 1.0f,1.0f,1.0f,1.0f };
        fbo.color.set(uv, color);
    }
}

CUDA void post(ivec2 NDC, FrameBufferGPU in, BuiltinRenderTargetGPU<RGBA> out) {
    constexpr float maxd = std::numeric_limits<unsigned int>::max();
    constexpr int rad = 5;
    constexpr auto base = 5.0f, sub =base*(rad*2+1)*(rad*2+1)-1.0f;
    float w = 0.0f;
    for (int i =  - rad; i <= rad; ++i)
        for (int j = - rad; j <=  rad; ++j)
            w +=in.depth.get({NDC.x+j,NDC.y+i })/maxd;
    w = w*base - sub*(in.depth.get(NDC)/maxd);
    auto c = in.color.get(NDC);
    NDC.y = in.mSize.y - NDC.y;
    out.set(NDC,c*w);
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo,DataViewer<Uniform> uniform,
    FrameBufferCPU& fbo, BuiltinRenderTargetGPU<RGBA> dest,Pipeline& pipeline) {
    fbo.colorRT->clear(pipeline,vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(pipeline);
    renderTriangles<VI, OI, Uniform, FrameBufferGPU, VS, drawPoint,setPoint>
        (pipeline,vbo, ibo, uniform, fbo.dataGPU,fbo.size);
    renderFullScreen<FrameBufferGPU, decltype(dest), post>(pipeline,fbo.dataGPU,dest,fbo.size);
}

