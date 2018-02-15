#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <PBR/Dist.hpp>

CUDAINLINE vec3 toPos(const vec3 p, const vec2 mul) {
    return {p.x * mul.x, p.y * mul.y, -p.z};
}

CUDAINLINE bool CS(unsigned int, vec3& pa, vec3& pb, vec3& pc, const Uniform& u) {
    pa = toPos(pa, u.mul);
    pb = toPos(pb, u.mul);
    pc = toPos(pc, u.mul);
    return true;
}

CUDAINLINE void VS(VI in, const Uniform& uniform, vec3& cpos, OI& out) {
    const auto wp = uniform.Msky * vec4(in.pos, 1.0f);
    out.get<Pos>() = in.pos;
    cpos = mat4(mat3(uniform.V)) * wp;
}

CUDAINLINE void setSkyPoint(unsigned int, ivec2 uv, float, const OI&, const OI&, const OI&,
    const Uniform&, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, 0xffffff00);
}

CUDAINLINE void drawSky(unsigned int, ivec2 uv, float, const OI& out, const OI&, const OI&, 
    const Uniform& uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) == 0xffffff00) {
        const vec3 p = out.get<Pos>();
        fbo.color.set(uv, uniform.sampler.getCubeMap(p));
        //fbo.color.set(uv, uniform.sampler.get(calcHDRUV(p)));
    }
}

CUDAINLINE void VSM(VI in, const Uniform& uniform, vec3& cpos, OI& out) {
    const auto wp = uniform.M * vec4(in.pos, 1.0f);
    out.get<Pos>() = wp;
    out.get<Normal>() = uniform.invM * in.normal;
    out.get<Tangent>() = uniform.invM * in.tangent;
    cpos = uniform.V * wp;
}

CUDAINLINE bool CSM(unsigned int id, vec3& pa, vec3& pb, vec3& pc, const Uniform& u) {
    pa = toPos(pa, u.mul);
    pb = toPos(pb, u.mul);
    pc = toPos(pc, u.mul);
    return u.cache.query(id);
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDAINLINE void setPoint(unsigned int, ivec2 uv, float z,const OI&,const OI&,const OI&, 
    const Uniform&, FrameBufferGPU& fbo) {
    fbo.depth.set(uv, z * maxdu);
}

CUDAINLINE void drawPoint(unsigned int triID, ivec2 uv, float z, const OI& out, const OI&,
    const OI&, const Uniform& uniform, FrameBufferGPU& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu)) {
        uniform.cache.record(triID);
        const vec3 p = out.get<Pos>();
        const vec3 N = normalize(out.get<Normal>());
        const vec3 X = normalize(out.get<Tangent>());
        const auto Y = normalize(cross(N, X));
        const auto off = uniform.lp - p;
        const auto dis2 = length2(off);
        const auto dis = sqrt(dis2);
        const auto L = off / dis;
        const auto V = normalize(uniform.cp - p);
        const auto F = disneyBRDF(L, V, N, X, Y, uniform.arg);
        const auto ref = reflect(-V, N);
        const auto lc = uniform.lc + 100.0f * vec3(uniform.sampler.getCubeMap(ref));
        const auto res = lc * F * (distUE4(dis2, uniform.r * uniform.r) * dot(N, L));
        fbo.color.set(uv, {res, 1.0f});
    }
}

CUDAINLINE void post(ivec2 NDC, const PostUniform& uni, BuiltinRenderTargetGPU<RGBA> out) {
    RGB c = uni.in.color.get(NDC);
    const auto lum = luminosity(c);
    if (uni.in.depth.get(NDC) < 0xffffff00) {
        if (lum > 0.0f) {
            atomicAdd(&uni.sum->first, log(lum));
            atomicInc(&uni.sum->second, maxv);
        }
        c = ACES(c, *uni.lum);
    }
    c = pow(c, vec3(1.0f / 2.2f));
    NDC.y = uni.in.mSize.y - 1 - NDC.y;
    out.set(NDC, {c, 1.0f});
}

CALLABLE void updateLum(PostUniform uniform) {
    *uniform.lum = calcLum(uniform.sum->first / (uniform.sum->second + 1));
}

template <VSF<VI, OI, Uniform> vs, TCSF<Uniform> cs, FSF<OI, Uniform, FrameBufferGPU> ds,
          FSF<OI, Uniform, FrameBufferGPU> fs>
void renderMesh(const StaticMesh& model, const MemoryRef<Uniform>& uniform,
                FrameBufferCPU& fbo, const Camera::RasterPosConverter converter,
                const CullFace mode, TriangleRenderingHistory& history, CommandBuffer& buffer) {
    auto vert = calcVertex<VI, OI, Uniform, vs>(buffer, model.vert, uniform);
    renderTriangles<SharedIndex, OI, Uniform, FrameBufferGPU, cs, ds, fs>(buffer, vert,
                                                                          SharedIndex{model.index}, uniform,
                                                                          fbo.getData(buffer), fbo.size,
                                                                          converter.near, converter.far, history, mode);
}

void kernel(const StaticMesh& model, TriangleRenderingHistory& mh,
            const StaticMesh& skybox, TriangleRenderingHistory& sh,
            const MemoryRef<Uniform>& uniform, FrameBufferCPU& fbo, float* lum,
            const Camera::RasterPosConverter converter, CommandBuffer& buffer) {
    fbo.colorRT->clear(buffer, vec4{0.0f, 0.0f, 0.0f, 1.0f});
    fbo.depthBuffer->clear(buffer);
    renderMesh<VSM, CSM, setPoint, drawPoint>(model, uniform, fbo, converter, CullFace::Back, mh, buffer);
    renderMesh<VS, CS, setSkyPoint, drawSky>(skybox, uniform, fbo, converter, CullFace::Front, sh, buffer);
    auto puni = buffer.allocConstant<PostUniform>();
    auto sum = buffer.allocBuffer<std::pair<float, unsigned int>>();
    buffer.memset(sum);
    auto punidata = buffer.makeLazyConstructor<PostUniform>(fbo.data, lum, sum);
    buffer.memcpy(puni, [punidata,&buffer](auto call) {
        auto pd = punidata;
        auto data = pd.get(buffer);
        call(&data);
    });
    const std::shared_ptr<Resource<BuiltinRenderTargetGPU<RGBA>>> res =
        std::make_shared<ImageResource>(buffer, fbo.image);
    const ResourceRef<BuiltinRenderTargetGPU<RGBA>> image{res};
    renderFullScreen<PostUniform, BuiltinRenderTargetGPU<RGBA>, post>(buffer, puni, image,
                                                                      fbo.size);
    buffer.callKernel(updateLum, punidata);
}
