#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <PBR/BRDF.hpp>
#include <device_functions.h>

CUDA void VS(VI in, Uniform uniform, vec4& NDC, OI& out) {
    auto wp =uniform.M*vec4(in.pos, 1.0f);
    out.get<pos>() = wp;
    out.get<normal>() = uniform.invM*in.normal;
    //out.get<bin>() = uniform.invM*in.tangent;
    NDC = uniform.VP*wp;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDA void setPoint(ivec2 uv,float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z*maxdu);
}

CUDA void drawPoint(ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) ==static_cast<unsigned int>(z*maxdu)) {
        auto p = out.get<pos>();
        vec3 nd =normalize(out.get<normal>());
        //auto tangent = normalize(out.get<bin>());
        auto in = uniform.dir;
        auto out = normalize(uniform.cp - p);
        auto h = calcHalf(in, out);
        auto ndi = dot(nd, in);
        auto ndo = dot(nd, out);
        auto idh = dot(in, h);
        auto diff = disneyDiffuse(ndi,ndo,idh,uniform.roughness);
        auto D = GGXD(ndo, uniform.roughness*uniform.roughness);
        auto F = fresnelSchlick(uniform.f0,fmax(idh,0.0f));
        auto G = smithG(ndi, ndo, 0.5f + 0.5f*uniform.roughness);
        auto w = cookTorrance(diff, D, F, G, ndi, ndo);
        auto res = uniform.color*uniform.lc*w*fmax(ndi,0.0f);
        //auto res = uniform.sampler.get(in,out,h,nd,tangent)*uniform.lc;
        fbo.color.set(uv, {res,1.0f });
    }
}

CUDA void post(ivec2 NDC, PostUniform uni, BuiltinRenderTargetGPU<RGBA> out) {
    RGB c = uni.in.color.get(NDC);
    auto lum = luminosity(c);
    if(lum>0.0f)atomicAdd(uni.sum,log(lum));
    c = ACES(c,uni.lum);
    NDC.y = uni.in.mSize.y- 1 - NDC.y;
    out.set(NDC, { c,1.0f });
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo, const Uniform* uniform
    , FrameBufferCPU& fbo, const PostUniform* puni,
    BuiltinRenderTargetGPU<RGBA> dest, Pipeline& pipeline) {
    fbo.colorRT->clear(pipeline,vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(pipeline);
    auto vert = calcVertex<VI,OI,Uniform,VS>(pipeline,vbo,uniform,fbo.size);
    renderTriangles<SharedIndex, OI, Uniform, FrameBufferGPU,setPoint , drawPoint>
        (pipeline, vert, ibo, uniform, fbo.dataGPU.get(), fbo.size);
    renderFullScreen<PostUniform, decltype(dest), post>(pipeline,puni,dest,fbo.size);
}

