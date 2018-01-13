#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <PBR/Dist.hpp>

CUDAInline void VS(VI in, Uniform uniform, vec3& cpos, OI& out) {
    auto wp =uniform.M*vec4(in.pos, 1.0f);
    out.get<texCoord>() = in.uv;
    out.get<pos>() = in.pos;
    //out.get<normal>() =uniform.invM*in.normal;
    //out.get<tangent>() =uniform.invM*in.tangent;
    cpos = uniform.V*wp;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDAInline void setPoint(unsigned int triID,ivec2 uv,float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z*maxdu);
}

CUDAInline void drawPoint(unsigned int triID,ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) ==static_cast<unsigned int>(z*maxdu)) {
        auto p = out.get<pos>();
        fbo.color.set(uv, uniform.sampler.getCubeMap(normalize(p)));
        /*
        ++triID;
        auto r = triID % 3;
        triID /= 3;
        auto g = triID % 3;
        triID /= 3;
        auto b = triID % 3;
        fbo.color.set(uv, vec4(r/3.0f,g/3.0f,b/3.0f,1.0f));
        */
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
        /*
        auto tex = out.get<texCoord>();
        tex *= 10.0f;
        auto p = (fmod(tex.x, 1.0f) > 0.5f) ^ (fmod(tex.y, 1.0f) < 0.5f);
        fbo.color.set(uv,vec4(p));
        */
    }
}

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

void kernel(const DataViewer<VI>& vbo,const DataViewer<uvec3>& ibo,
    const MemoryRef<Uniform>& uniform, FrameBufferCPU & fbo, float* lum,
    Camera::RasterPosConverter converter,CommandBuffer & buffer) {
    fbo.colorRT->clear(buffer, vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(buffer);
    auto vert = calcVertex<VI, OI, Uniform, VS>(buffer, vbo, uniform);
    renderTriangles<SharedIndex, OI, Uniform, FrameBufferGPU, setPoint, drawPoint>(buffer, vert, 
        ibo, uniform,fbo.getData(buffer), fbo.size,converter.near,converter.far,converter.mul,CullFace::None);
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
    renderFullScreen<PostUniform,BuiltinRenderTargetGPU<RGBA>, post>(buffer, puni, image, 
        fbo.size);
    buffer.callKernel(updateLum,punidata);
}
