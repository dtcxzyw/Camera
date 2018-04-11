#include <PostProcess/ToneMapping.hpp>
#include <Rasterizer/PostProcess.hpp>
#include "kernel.hpp"
#include <PBR/Dist.hpp>
#include <Rasterizer/SphereRasterizer.hpp>
#include <Rasterizer/IndexDescriptor.hpp>

CUDAINLINE vec3 toPos(const vec3 p, const Uniform& u) {
    return {p.x * u.mul.x, p.y * u.mul.y, -p.z};
}

CUDAINLINE void setDepth(unsigned int& data, const unsigned int val) {
    atomicMin(&data, val);
}

CUDAINLINE bool CS(unsigned int, vec3& pa, vec3& pb, vec3& pc, const Uniform& u) {
    pa = toPos(pa, u);
    pb = toPos(pb, u);
    pc = toPos(pc, u);
    return true;
}

CUDAINLINE void VS(VI in, const Uniform& uniform, vec3& cpos, OI& out) {
    const auto wp = uniform.Msky * vec4(in.pos, 1.0f);
    out.get<Pos>() = in.pos;
    cpos = mat4(mat3(uniform.V)) * wp;
}

CUDAINLINE void drawSky(unsigned int, ivec2 uv, float, const OI& out, const OI&, const OI&,
    const Uniform& uniform, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == 0xffffffff) {
        const vec3 p = out.get<Pos>();
        fbo.color.set(uv, uniform.sampler.getCubeMap(p));
    }
}

CUDAINLINE void VSM(VI in, const Uniform& uniform, vec3& cpos, OI& out) {
    const auto wp = uniform.M * vec4(in.pos, 1.0f);
    out.get<Pos>() = wp;
    out.get<Normal>() = uniform.normalMat * in.normal;
    out.get<Tangent>() = uniform.normalMat * in.tangent;
    cpos = uniform.V * wp;
}

CUDAINLINE bool CSM(unsigned int, vec3& pa, vec3& pb, vec3& pc, const Uniform& u) {
    pa = toPos(pa, u);
    pb = toPos(pb, u);
    pc = toPos(pc, u);
    return true;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

CUDAINLINE void setModel(unsigned int, ivec2 uv, float z, const OI&, const OI&, const OI&,
    const Uniform&, FrameBufferRef& fbo) {
    setDepth(fbo.depth.get(uv), z * maxdu);
}

CUDAINLINE void drawModel(unsigned int, ivec2 uv, float z, const OI& out, const OI&,
    const OI&, const Uniform& uniform, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu)) {
        const vec3 p = out.get<Pos>();
        const vec3 N = normalize(out.get<Normal>());
        const vec3 T = normalize(out.get<Tangent>());
        const auto X = normalize(T - dot(T, N) * N);
        const auto Y = normalize(cross(X, N));
        const auto off = uniform.lp - p;
        const auto dis2 = length2(off);
        const auto dis = sqrt(dis2);
        const auto L = off / dis;
        const auto V = normalize(uniform.cp - p);
        const auto F = disneyBRDF(L, V, N, X, Y, uniform.arg);
        const auto ref = reflect(-V, N);
        const auto lc = uniform.lc * distUE4(dis2, uniform.r2) +
            vec3(uniform.sampler.getCubeMap(ref));
        const auto res = lc * F * fabs(dot(N, L));
        fbo.color.set(uv, {res, 1.0f});
    }
}

CUDAINLINE void setPoint(unsigned int, ivec2 uv, float z, const OI&,
    const Uniform&, FrameBufferRef& fbo) {
    setDepth(fbo.depth.get(uv), z * maxdu);
}

CUDAINLINE void drawPoint(unsigned int, ivec2 uv, float z, const OI&,
    const Uniform&, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu))
        fbo.color.set(uv, {1.0f, 1.0f, 1.0f, 1.0f});
}

CUDAINLINE void post(ivec2 NDC, const PostUniform& uni, BuiltinRenderTargetRef<RGBA8> out) {
    RGB c = uni.in.color.get(NDC);
    const auto lum = luminosity(c);
    if (uni.in.depth.get(NDC) < 0xffffffff) {
        if (lum > 0.0f) {
            atomicAdd(&uni.sum->first, log(lum));
            atomicInc(&uni.sum->second, maxv);
        }
        c = ACES(c, *uni.lum);
    }
    c = clamp(pow(c, vec3(1.0f / 2.2f)), 0.0f, 1.0f);
    const RGBA8 color = {c * 255.0f, 255};
    out.set(NDC, color);
}

GLOBAL void updateLum(const PostUniform uniform) {
    *uniform.lum = calcLum(uniform.sum->first / (uniform.sum->second + 1));
}

template <VertShader<VI, OI, Uniform> VertFunc, TriangleClipShader<Uniform> ClipFunc, 
    FragmentShader<OI, Uniform, FrameBufferRef>... FragFunc>
void renderMesh(const StaticMesh& model, const Span<Uniform>& uniform,
    const Span<FrameBufferRef>& frameBuffer, const uvec2 size,
    const Camera::RasterPosConverter converter,
    const CullFace mode, RenderingContext& context, const vec4 scissor,
    CommandBuffer& buffer) {
    auto vert = calcVertex<VI, OI, Uniform, VertFunc>(buffer, buffer.useAllocated(model.vert),
        uniform, context.get());
    const auto index = makeIndexDescriptor<SeparateTrianglesWithIndex>(model.index.size(),
        buffer.useAllocated(model.index));
    renderTriangles<decltype(index), OI, Uniform, FrameBufferRef, ClipFunc,
        emptyTriangleTileClipShader<Uniform>, VersionComparer, FragFunc...>(buffer, vert, index,
            uniform, frameBuffer, size, converter.near, converter.far, scissor, context.triContext, 
            context.vertCounter.get(), mode);
}

CUDAINLINE vec4 vsSphere(vec4 sp, const Uniform& uniform) {
    return calcCameraSphere(sp, uniform.V);
}

CUDAINLINE void setSpherePoint(unsigned int, ivec2 uv, float z, vec3, vec3, float, bool,
    vec2, const Uniform&, FrameBufferRef& fbo) {
    setDepth(fbo.depth.get(uv), z * maxdu);
}

CUDAINLINE void drawSpherePoint(unsigned int, ivec2 uv, float z, vec3 p, vec3 dir, float invr,
    bool inSphere, vec2, const Uniform& u, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu)) {
        const vec3 pos = u.invV * vec4(p, 1.0f);
        const auto normalizedDir = u.normalInvV * dir * invr;
        const auto N = calcSphereNormal(normalizedDir, inSphere);
        const auto Y = calcSphereBiTangent(N);
        const auto X = calcSphereTangent(N, Y);
        const auto off = u.lp - pos;
        const auto dis2 = length2(off);
        const auto dis = sqrt(dis2);
        const auto L = off / dis;
        const auto V = normalize(u.cp - pos);
        const auto F = disneyBRDF(L, V, N, X, Y, u.arg);
        const auto ref = reflect(-V, N);
        const auto lc = u.lc * distUE4(dis2, u.r2) + vec3(u.sampler.getCubeMap(ref));
        const auto res = lc * F * fabs(dot(N, L));
        fbo.color.set(uv, {res, 1.0f});
    }
}

void kernel(const StaticMesh& model, RenderingContext& mc,
    const StaticMesh& skybox, RenderingContext& sc,
    const MemorySpan<vec4>& spheres,
    const Span<Uniform>& uniform, FrameBuffer& fbo, float* lum,
    const Camera::RasterPosConverter converter, CommandBuffer& buffer) {
    fbo.colorRT->clear(buffer, vec4{ 0.0f, 0.0f, 0.0f, 1.0f });
    Buffer2D<unsigned int> depth(buffer, fbo.size);
    depth.clear(0xff);
    const auto frameBuffer = fbo.getData(buffer, depth);
    const vec4 scissor = {0.0f, fbo.size.x, 0.0f, fbo.size.y};
    renderMesh<VSM, CSM, setModel, drawModel>(model, uniform, frameBuffer, fbo.size,
        converter, CullFace::Back, mc, scissor, buffer);
    renderSpheres<Uniform, FrameBufferRef, vsSphere, setSpherePoint, drawSpherePoint>(buffer,
        buffer.useAllocated(spheres), uniform, frameBuffer, fbo.size, converter.near, converter.far, 
        converter.mul, scissor);
    renderMesh<VS, CS, drawSky>(skybox, uniform, frameBuffer, fbo.size, converter,
        CullFace::Front, sc, scissor, buffer);
    const auto puni = buffer.allocConstant<PostUniform>();
    const auto sum = buffer.allocBuffer<std::pair<float, unsigned int>>();
    buffer.memset(sum);
    const auto depthBufferRef = depth.toBuffer();
    const auto fboData = buffer.makeLazyConstructor<FrameBufferRef>(fbo.data, depthBufferRef);
    auto punidata = buffer.makeLazyConstructor<PostUniform>(fboData, lum, sum);
    auto&& manager = buffer.getResourceManager();
    buffer.memcpy(puni, [punidata,&manager](auto call) {
        auto pd = punidata;
        auto data = pd.get(manager);
        call(&data);
    });
    renderFullScreen<PostUniform, BuiltinRenderTargetRef<RGBA8>, post>(buffer, puni,
        fbo.postRT->toTarget(), fbo.size);
    buffer.callKernel(updateLum, punidata);
}
