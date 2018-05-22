#include <PostProcess/ToneMapping.hpp>
#include <Rasterizer/PostProcess.hpp>
#include "kernel.hpp"
#include <Rasterizer/SphereRasterizer.hpp>
#include <Rasterizer/IndexDescriptor.hpp>
#include <Texture/Noise.hpp>

DEVICEINLINE Point toPos(const Point pos, const Uniform& u) {
    const Vector p{ pos };
    return {p.x * u.mul.x, p.y * u.mul.y, -p.z};
}

DEVICEINLINE void setDepth(unsigned int& data, const unsigned int val) {
    atomicMin(&data, val);
}

DEVICEINLINE bool CS(unsigned int, Point& pa, Point& pb, Point& pc, const Uniform& u) {
    pa = toPos(pa, u);
    pb = toPos(pb, u);
    pc = toPos(pc, u);
    return true;
}

DEVICEINLINE void VS(VI in, const Uniform& uniform, Point& cpos, OI& out) {
    out.get<Pos>() = in.pos;
    cpos = uniform.skyTransform(in.pos);
}

DEVICEINLINE void drawSky(unsigned int, ivec2 uv, float, const OI& out, const OI&, const OI&,
    const Uniform& uniform, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == 0xffffffff) {
        const auto p = out.get<Pos>();
        fbo.color.set(uv, uniform.sampler.getCubeMap(Vector(p)));
    }
}

DEVICEINLINE void VSM(VI in, const Uniform& uniform, Point& cpos, OI& out) {
    const auto wp = uniform.modelTransform(in.pos);
    out.get<Pos>() = wp;
    out.get<Nor>() = Vector(uniform.modelTransform(makeNormalUnsafe(in.normal)));
    out.get<Tangent>() = Vector(uniform.modelTransform(makeNormalUnsafe(in.tangent)));
    cpos = uniform.cameraTransform(wp);
}

DEVICEINLINE bool CSM(unsigned int, Point& pa, Point& pb, Point& pc, const Uniform& u) {
    pa = toPos(pa, u);
    pb = toPos(pb, u);
    pc = toPos(pc, u);
    return true;
}

constexpr float maxdu = std::numeric_limits<unsigned int>::max();

DEVICEINLINE void setModel(unsigned int, ivec2 uv, float z, const OI&, const OI&, const OI&,
    const Uniform&, FrameBufferRef& fbo) {
    setDepth(fbo.depth.get(uv), z * maxdu);
}

DEVICEINLINE Spectrum shade(const Point p, const Normal N, const Normal X, const Normal Y,
    const Uniform& uniform) {
    const auto sample = uniform.light.sampleLi({}, p);
    const Normal V{ uniform.cp - p };
    const auto F = disneyBRDF(Normal(sample.wi), V, N, X, Y, uniform.arg);
    const auto ref = reflect(-V, N);
    const auto lc = sample.illumination +
        Spectrum(uniform.sampler.getCubeMap(Vector(ref)));
    return lc * F * fabs(dot(N, sample.wi));
}

DEVICEINLINE void drawModel(unsigned int, ivec2 uv, float z, const OI& out, const OI& ddx,
    const OI& ddy, const Uniform& uniform, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu)) {
        const Point p = out.get<Pos>();
        const Normal N{ out.get<Nor>() };
        Normal X{ out.get<Tangent>() };
        const auto Y = cross(X, N);
        X = cross(Y, N);
        const auto octaves = calcOctavesAntiAliased(Vector(ddx.get<Pos>()), Vector(ddy.get<Pos>()));
        const auto col = marble(Vector(N)*30.0f, 1.0f, 0.8f, octaves) + Spectrum(0.5f);
        fbo.color.set(uv, { (shade(p,N,X,Y,uniform)*col).toRGB(), 1.0f });
    }
}

DEVICEINLINE void setPoint(unsigned int, ivec2 uv, float z, const OI&,
    const Uniform&, FrameBufferRef& fbo) {
    setDepth(fbo.depth.get(uv), z * maxdu);
}

DEVICEINLINE void drawPoint(unsigned int, ivec2 uv, float z, const OI&,
    const Uniform&, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu))
        fbo.color.set(uv, {1.0f, 1.0f, 1.0f, 1.0f});
}

DEVICEINLINE void post(ivec2 NDC, const PostUniform& uni, BuiltinRenderTargetRef<RGBA8> out) {
    Spectrum c(uni.in.color.get(NDC));
    if (uni.in.depth.get(NDC) < 0xffffffff) {
        const auto lum = c.lum();
        if (lum > 0.0f) {
            atomicAdd(&uni.sum->first, log(lum));
            atomicInc(&uni.sum->second, maxv);
        }
        c = Spectrum(ACES(c.toRGB(), *uni.lum));
    }
    const RGBA8 color = { clamp(pow(c.toRGB(),RGB(1.0f / 2.2f)), 0.0f, 1.0f) * 255.0f, 255 };
    out.set(NDC, color);
}

GLOBAL void updateLum(const PostUniform uniform) {
    *uniform.lum = calcLum(uniform.sum->first / (uniform.sum->second + 1));
}

template <VertShader<VI, OI, Uniform> VertFunc, TriangleClipShader<Uniform> ClipFunc, 
    FragmentShader<OI, Uniform, FrameBufferRef>... FragFunc>
void renderMesh(const StaticMeshData& model, const Span<Uniform>& uniform,
    const Span<FrameBufferRef>& frameBuffer, const uvec2 size,
    const PinholeCamera::RasterPosConverter converter,
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

DEVICEINLINE SphereDesc vsSphere(SphereDesc sp, const Uniform& uniform) {
    return calcCameraSphere(uniform.cameraTransform, sp);
}

DEVICEINLINE void setSpherePoint(unsigned int, ivec2 uv, float z, Point, Vector, float, bool,
    Vector, Vector, const Uniform&, FrameBufferRef& fbo) {
    setDepth(fbo.depth.get(uv), z * maxdu);
}

DEVICEINLINE void drawSpherePoint(unsigned int, ivec2 uv, float z, Point p, Vector dir, float invr,
    bool inSphere, Vector dpdx, Vector dpdy, const Uniform& u, FrameBufferRef& fbo) {
    if (fbo.depth.get(uv) == static_cast<unsigned int>(z * maxdu)) {
        const auto pos = u.invCameraTransform(p);
        const auto modelDir = u.invCameraTransform(dir);
        const auto normalizedDir = makeNormalUnsafe(modelDir * invr);
        const auto N = calcSphereNormal(normalizedDir, inSphere);
        const auto Y = calcSphereBiTangent(N);
        const auto X = calcSphereTangent(N, Y);
        auto res = shade(pos, N, X, Y, u);
        const auto octaves = calcOctavesAntiAliased(u.invCameraTransform(dpdx),
            u.invCameraTransform(dpdy));
        res *= marble(modelDir, 1.0f, 0.5f, octaves) + Spectrum(0.5f);
        fbo.color.set(uv, { res.toRGB(), 1.0f });
    }
}

void kernel(const StaticMeshData& model, RenderingContext& mc,
    const StaticMeshData& skybox, RenderingContext& sc,
    const MemorySpan<SphereDesc>& spheres,
    const Span<Uniform>& uniform, FrameBuffer& fbo, float* lum,
    const PinholeCamera::RasterPosConverter converter, CommandBuffer& buffer) {
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
        fbo.postRT->toRef(), fbo.size);
    buffer.callKernel(updateLum, punidata);
}
