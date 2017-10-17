#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <PBR/PBRFunctions.hpp>

CUDA void VS(VI in, Uniform uniform, vec4& NDC, OI& out) {
    auto wp =uniform.M*vec4(in.pos, 1.0f);
    out.get<pos>() = wp;
    out.get<normal>() = in.normal;
    NDC = uniform.VP*wp;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDA void setPoint(ivec2 uv,float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z*maxdu);
}

CUDA void drawPoint(ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) ==static_cast<unsigned int>(z*maxdu)) {
        auto p = out.get<pos>();
        vec3 nd = out.get<normal>();
        auto in = uniform.dir;
        auto out = normalize(uniform.cp - p);
        auto h = calcHalf(in, out);
        auto ndi = dot(nd, in);
        auto ndo = dot(nd, out);
        auto idh = dot(in, h);
        vec4 c = { 1.0f,1.0f,1.0f,1.0f };
        auto diff = diffuse(ndi,ndo,idh,uniform.roughness);
        
    }
}

CUDA void post(ivec2 NDC, FrameBufferGPU in, BuiltinRenderTargetGPU<RGBA> out) {
    auto c = in.color.get(NDC);
    NDC.y = in.mSize.y-1 - NDC.y;
    out.set(NDC, c);
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo,DataViewer<Uniform> uniform,
    FrameBufferCPU& fbo, BuiltinRenderTargetGPU<RGBA> dest,Pipeline& pipeline) {
    fbo.colorRT->clear(pipeline,vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(pipeline);
    renderTriangles<VI, OI, Uniform, FrameBufferGPU, VS, drawPoint,setPoint>
        (pipeline,vbo, ibo, uniform, fbo.dataGPU,fbo.size);
    renderFullScreen<FrameBufferGPU, decltype(dest), post>(pipeline,fbo.dataGPU,dest,fbo.size);
}

