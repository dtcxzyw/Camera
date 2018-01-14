#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <PBR/Dist.hpp>

CUDAInline void VS(VI in, Uniform uniform, vec3& cpos, OI& out) {
    auto wp =uniform.M*vec4(in.pos, 1.0f);
    out.get<pos>() = in.pos;
    cpos = uniform.V*wp;
}

CUDAInline void VSM(VI in, Uniform uniform, vec3& cpos, OI& out) {
    auto wp = uniform.M*vec4(in.pos, 1.0f);
    out.get<pos>() = wp;
    out.get<normal>() =uniform.invM*in.normal;
    out.get<tangent>() =uniform.invM*in.tangent;
    cpos = uniform.V*wp;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDAInline void setPoint(unsigned int triID,ivec2 uv,float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z*maxdu);
}

CUDAInline void drawSky(unsigned int triID,ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) ==static_cast<unsigned int>(z*maxdu)) {
        auto p = out.get<pos>();
        fbo.color.set(uv, uniform.sampler.getCubeMap(normalize(p)));
    }
}

CUDAInline void drawPoint(unsigned int triID, ivec2 uv, float z, OI out, Uniform uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z*maxdu)) {
        auto p = out.get<pos>();
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
        fbo.color.set(uv, { res,1.0f });
    }
}

CUDAInline void post(ivec2 NDC, PostUniform uni, BuiltinRenderTargetGPU<RGBA> out) {
    RGB c = uni.in.color.get(NDC);
    auto lum = luminosity(c);
    if (lum > 1e-3f) {
        atomicAdd(&uni.sum->first, log(lum));
        atomicInc(&uni.sum->second, maxv);
    }
    c = pow(ACES(c, *uni.lum), vec3(1.0f / 2.2f));
    NDC.y = uni.in.mSize.y - 1 - NDC.y;
    out.set(NDC, { c,1.0f });
}

CALLABLE void updateLum(PostUniform uniform) {
    *uniform.lum=calcLum(uniform.sum->first/(uniform.sum->second+1));
}

template<VSF<VI,OI,Uniform> vs,FSF<OI,Uniform,FrameBufferGPU> fs>
void renderMesh(const StaticMesh& model , const MemoryRef<Uniform>& uniform,
    FrameBufferCPU & fbo, Camera::RasterPosConverter converter,CullFace mode,CommandBuffer & buffer) {
    auto vert = calcVertex<VI, OI, Uniform, vs>(buffer, model.mVert, uniform);
    renderTriangles<SharedIndex, OI, Uniform, FrameBufferGPU, setPoint, fs>(buffer, vert,
        model.mIndex, uniform, fbo.getData(buffer), fbo.size,
        converter.near, converter.far,converter.mul,mode);
}

void kernel(const StaticMesh& model, const StaticMesh& skybox,
    const MemoryRef<Uniform>& uniform, FrameBufferCPU & fbo, float* lum,
    Camera::RasterPosConverter converter, CommandBuffer & buffer) {
    fbo.colorRT->clear(buffer, vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(buffer);
    renderMesh<VSM,drawPoint>(model,uniform,fbo,converter,CullFace::Back,buffer);
    renderMesh<VS, drawSky>(skybox, uniform, fbo, converter,CullFace::Front, buffer);
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
