#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <ScanLineRenderer/Primitive.hpp>
#include <PBR/Dist.hpp>

/*
CUDA void GS(VI* in, Uniform uniform, Queue<VI, 2> out) {
    auto dab = in[0].pos - in[1].pos, dcb = in[2].pos - in[1].pos;
    auto off = normalize(cross(dcb, dab))*uniform.off;
    auto p = (in[0].pos + in[1].pos + in[2].pos) / 3.0f;
    in[0].pos = p;
    in[1].pos = p + off;
    out.push(in);
}
*/

CUDAInline void VS(VI in, Uniform uniform, vec4& NDC, OI& out) {
    auto wp =uniform.M*vec4(in.pos, 1.0f);
    out.get<texCoord>() = in.uv;
    out.get<pos>() = in.pos;
    //out.get<normal>() =uniform.invM*in.normal;
    //out.get<tangent>() =uniform.invM*in.tangent;
    NDC = uniform.PV*wp;
    //NDC.z = NDC.w*0.999f;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDAInline void setPoint(ivec2 uv,float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z*maxdu);
}

CUDAInline void drawPoint(ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) ==static_cast<unsigned int>(z*maxdu)) {
        //auto p = out.get<pos>();
        auto tex = out.get<texCoord>();
        /*
        vec3 N =normalize(out.get<normal>());
        vec3 X = normalize(out.get<tangent>());
        vec3 Y = normalize(cross(N, X));
        auto off = uniform.lp - p;
        auto dis2 = length2(off);
        auto dis = sqrt(dis2);
        auto L = off/dis;
        auto V = normalize(uniform.cp - p);
        auto F = disneyBRDF(L, V, N, X, Y, uniform.arg);
        auto res = uniform.lc*F*(distUE4(dis2,uniform.r*uniform.r)*dot(N, L));
        */
        tex *= 10.0f;
        auto p = (fmod(tex.x, 1.0f) > 0.5f) ^ (fmod(tex.y, 1.0f) < 0.5f);
        fbo.color.set(uv,vec4(p));
    }
}

/*
CUDAInline void drawHair(ivec2 uv, float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z*maxdu))
        fbo.color.set(uv, vec4(1.0f));
}
*/

CUDAInline void post(ivec2 NDC, PostUniform uni, BuiltinRenderTargetGPU<RGBA> out) {
    RGB c = uni.in.color.get(NDC);
    auto lum = luminosity(c);
    if (lum > 0.0f) {
        atomicAdd(&uni.sum->first, fmin(log(lum), 5.0f));
        atomicInc(&uni.sum->second, maxv);
    }
    c = pow(ACES(c, *uni.lum), vec3(1.0f / 2.2f));
    NDC.y = uni.in.mSize.y - 1 - NDC.y;
    out.set(NDC, { c,1.0f });
}

CALLABLE void updateLum(PostUniform uniform) {
    *uniform.lum=calcLum(uniform.sum->first/(uniform.sum->second+1));
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo, const MemoryRef<Uniform>& uniform
    , FrameBufferCPU & fbo, float* lum,CommandBuffer & buffer) {
    fbo.colorRT->clear(buffer, vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(buffer);
    auto vert = calcVertex<VI, OI, Uniform, VS>(buffer, vbo, uniform, fbo.size);
    renderTriangles<SharedIndex, OI, Uniform, FrameBufferGPU, setPoint, drawPoint>
        (buffer, vert, ibo, uniform,fbo.getData(buffer), fbo.size);
    //auto prim = genPrimitive<3,2,SharedIndex, VI, Uniform, GS>(stream, vbo,ibo,uniform,ibo.size());
    //auto pv= calcVertex<VI, OI, Uniform, VS>(stream, prim, uniform, fbo.size);
    //renderLines<OI, Uniform, FrameBufferGPU, setPoint, drawHair>(stream, pv, uniform
    //   , fbo.dataGPU.get(), fbo.size);
    auto puni = buffer.allocConstant<PostUniform>();
    auto sum = buffer.allocBuffer<std::pair<float, unsigned int>>();
    buffer.memset(sum);
    auto punidata = buffer.makeLazyConstructor<PostUniform>(fbo.data,lum,sum);
    buffer.memcpy(puni, [punidata,&buffer](auto call) {
        auto pd = punidata;
        auto data=pd.get(buffer);
        call(&data);
    });
    ResourceRef<BuiltinRenderTargetGPU<RGBA>> image
        = std::make_shared<ImageResource>(buffer,fbo.image);
    renderFullScreen<PostUniform,BuiltinRenderTargetGPU<RGBA>, post>(buffer, puni
        , image, fbo.size);
    buffer.callKernel(updateLum,punidata);
}
