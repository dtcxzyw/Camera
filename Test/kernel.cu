#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <PBR/BRDF.hpp>
#include <device_functions.h>

CUDA void VS(VI in, Uniform uniform, vec4& NDC, OI& out) {
    auto wp =uniform.M*vec4(in.pos, 1.0f);
    out.get<pos>() = wp;
    NDC = uniform.VP*wp;
    out.get<normal>() =uniform.invM*in.normal;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDA void setPoint(ivec2 uv,float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z*maxdu);
}

CUDA void drawPoint(ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) ==static_cast<unsigned int>(z*maxdu)) {
        auto p = out.get<pos>();
        vec3 nd =normalize(out.get<normal>());
        auto in = uniform.dir;
        auto out = normalize(uniform.cp - p);
        auto h = calcHalf(in, out);
        auto ndi = dot(nd, in);
        auto ndo = dot(nd, out);
        auto idh = dot(in, h);
        auto diff = disneyDiffuse(ndi,ndo,idh,uniform.roughness);
        auto G = smithG(ndi,ndo,0.5f+0.5f*uniform.roughness);
        auto F = fresnelSchlick(uniform.f0,fmax(idh,0.0f));
        auto D = GGXD(ndo, uniform.roughness*uniform.roughness);
        auto wSpec = cookTorrance(diff, D, F, G, ndi, ndo);
        auto wDiff = (1.0f - F)*(1.0f-uniform.metallic)*uniform.albedo*one_over_pi<float>();
        auto w = (wSpec + wDiff)*uniform.lc*fmax(ndi,0.0f);
        auto res = uniform.color*w;
        res = convertLinearToSRGB(res);
        fbo.color.set(uv, {res,1.0f });
    }
}

CUDA void post(ivec2 NDC, FrameBufferGPU in, BuiltinRenderTargetGPU<RGBA> out) {
    vec3 c = in.color.get(NDC);
    NDC.y = in.mSize.y-1 - NDC.y;
    out.set(NDC, { c,1.0f });
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo,DataViewer<Uniform> uniform,
    FrameBufferCPU& fbo, BuiltinRenderTargetGPU<RGBA> dest,Pipeline& pipeline) {
    fbo.colorRT->clear(pipeline,vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(pipeline);
    renderTriangles<VI, OI, Uniform, FrameBufferGPU, VS, drawPoint,setPoint>
        (pipeline,vbo, ibo, uniform, fbo.dataGPU,fbo.size);
    renderFullScreen<FrameBufferGPU, decltype(dest), post>(pipeline,fbo.dataGPU,dest,fbo.size);
}

