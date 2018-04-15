#pragma once
#include <Rasterizer/Shared.hpp>
#include <Core/DispatchSystem.hpp>
#include <Rasterizer/Tile.hpp>

template <typename Uniform>
using SphereVertShader = vec4(*)(vec4 sp, const Uniform& uniform);

CUDAINLINE vec4 calcCameraSphere(const vec4 sp, const mat4& mat) {
    return {vec3{mat * vec4(sp.x, sp.y, sp.z, 1.0f)}, sp.w};
}

template <typename Uniform, SphereVertShader<Uniform> Func>
GLOBAL void calcCameraSpheres(const unsigned int size, READONLY(vec4) in, vec4* out,
                              READONLY(Uniform) uniform) {
    const auto id = getId();
    if (id >= size)return;
    out[id] = Func(in[id], *uniform);
}

/*
 *Utility Functions
 *x=sin theta cos phi
 *y=cos theta
 *z=sin theta sin phi
 *u=phi/2pi
 *v=theta/pi
 */

CUDAINLINE vec3 calcSphereNormal(const vec3 normalizedWorldDir, const bool inSphere) {
    return inSphere ? -normalizedWorldDir : normalizedWorldDir;
}

CUDAINLINE vec3 calcSphereBiTangent(const vec3 normal) {
    const auto left = normalize(cross({0.0f, 1.0f, 0.0f}, normal));
    return cross(left, normal);
}

CUDAINLINE vec3 calcSphereTangent(const vec3 normal, const vec3 biTangent) {
    return cross(biTangent, normal);
}

CUDAINLINE vec2 calcSphereTexCoord(const vec3 normalizedWorldDir) {
    const auto theta = std::acos(normalizedWorldDir.y);
    const auto phi = std::atan2(normalizedWorldDir.z, normalizedWorldDir.x) + pi<float>();
    return { phi * one_over_two_pi<float>(), theta * one_over_pi<float>() };
}

//dt.x/dx=dn.x/dx*dt.x/dn.x=dn.x/dx*1/(1+n.x*n.x)*(1/2pi)
//dt.y/dx=dn.y/dx*dt.y/dn.y=dn.y/dx*-sqrt(1-n.y^2)*(1/pi)
CUDAINLINE vec2 calcSphereTextureDerivative(const vec3 dir,const vec3 dndx) {
    return { dndx.x / (1 + dir.x*dir.x)*one_over_two_pi<float>(),
        dndx.y*-sqrt(1.0f - dir.y*dir.y)*one_over_pi<float>() };
}

struct STRUCT_ALIGN SphereInfo final {
    unsigned int id;
    vec4 info;
    float c;
};

struct SphereProcessingResult final {
    Span<unsigned int> offset;
    Span<SphereInfo> info;
    Span<TileRef> ref;

    SphereProcessingResult(const Span<unsigned int>& off,
                        const Span<SphereInfo>& sphereInfo, const Span<TileRef>& sphereRef)
        : offset(off), info(sphereInfo), ref(sphereRef) {}
};

SphereProcessingResult processSphereInfo(CommandBuffer& buffer,
    const Span<vec4>& spheres, vec4 scissor, vec2 hsiz, float near, float far, vec2 mul);

/*
 * pos=td
 * NDC.x=pos.x*mul.x/-pos.z
 * kx=pos.x/-pos.z=NDC.x/mul.x
 * NDC.y=pos.y*mul.y/-pos.z
 * ky=pos.y/-pos.z=NDC.y/mul.y
 * NDC.z=-pos.z
 * t=-pos.z [near,far]
 * d=(kx,ky,-1)
 * solve at^2+bt+c=0
 */

CUDAINLINE vec2 raster2NDC(const vec2 p, const float ihx, const float ihy) {
    return {p.x * ihx - 1.0f, 1.0f - p.y * ihy};
}

//in camera pos
template <typename Uniform, typename FrameBuffer>
using SphereFragmentShader = void(*)(unsigned int id, ivec2 uv, float z, vec3 pos, vec3 dir,
    float invr, bool inSphere, vec3 dpdx,vec3 dpdy, const Uniform& uniform, FrameBuffer& frameBuffer);

//2,4,8,16,32
template <typename Uniform, typename FrameBuffer,
    SphereFragmentShader<Uniform, FrameBuffer> FragShader>
GLOBAL void drawMicroS(READONLY(SphereInfo) info, READONLY(TileRef) idx,
                       READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
                       const float near, const float far, const float invnf, const float imx, const float imy,
                       const float ihx, const float ihy) {
    const auto ref = idx[blockIdx.x];
    const auto offX = threadIdx.x >> 1U, offY = threadIdx.x & 1U;
    const ivec2 uv{ref.rect.x + (threadIdx.y << 1) + offX, ref.rect.z + (threadIdx.z << 1) + offY};
    const vec2 p{uv.x + 0.5f, uv.y + 0.5f};
    if (p.x > ref.rect.y | p.y > ref.rect.w)return;
    const auto NDC = raster2NDC(p, ihx, ihy);
    const vec3 d = {NDC.x * imx, NDC.y * imy, -1.0f};
    const auto sphere = info[ref.id];
    const vec3 mid = sphere.info;
    const auto a = length2(d);
    const auto b = -2.0f * dot(d, mid);
    const auto c = sphere.c;
    const auto delta = b * b - 4.0f * a * c;
    if (delta < 0.0f)return;
    const auto sqrtDelta = sqrt(delta), inv2a = 0.5f / a;
    const auto t1 = (-sqrtDelta - b) * inv2a, t2 = (sqrtDelta - b) * inv2a;
    float t;
    bool inSphere;
    if (near <= t1 & t1 <= far)t = t1, inSphere = false;
    else if (near <= t2 & t2 <= far)t = t2, inSphere = true;
    else return;
    const auto pos = t * d;
    const auto dpdx = (shuffleVec3(pos, 0b10) - pos) * (offX ? -1.0f : 1.0f);
    const auto dpdy = (shuffleVec3(pos, 0b01) - pos) * (offY ? -1.0f : 1.0f);
    const auto nz = (t - near) * invnf;
    FragShader(sphere.id, p, nz, pos, pos - mid, sphere.info.w, inSphere, dpdx, dpdy, *uniform,
        *frameBuffer);
}

template <typename Uniform, typename FrameBuffer,
    SphereFragmentShader<Uniform, FrameBuffer> Func,
    SphereFragmentShader<Uniform, FrameBuffer>... Then>
CUDAINLINE void applySFS(unsigned int* offset, SphereInfo* info, TileRef* idx, Uniform* uniform,
                         FrameBuffer* frameBuffer, const float near, const float far, const float invnf,
                         const vec2 invMul, const vec2 invHsiz) {
    #pragma unroll
    for (auto i = 0; i < 5; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            dim3 grid(size);
            const auto bsiz = 1 << i;
            dim3 block(4, bsiz, bsiz);
            drawMicroS<Uniform, FrameBuffer, Func> << <grid, block >> >(info, idx + offset[i],
                                                                         uniform, frameBuffer, near, far, invnf,
                                                                         invMul.x, invMul.y, invHsiz.x, invHsiz.y);
        }
    }

    cudaDeviceSynchronize();
    applySFS<Uniform, FrameBuffer, Then...>(offset, info, idx, uniform, frameBuffer, near, far, invnf,
                                            invMul, invHsiz);
}

template <typename Uniform, typename FrameBuffer>
CUDAINLINE void applySFS(unsigned int*, SphereInfo*, TileRef*, Uniform*, FrameBuffer*,
                         const float, const float, const float, const vec2, const vec2) {}

template <typename Uniform, typename FrameBuffer,
    SphereFragmentShader<Uniform, FrameBuffer>... FragShader>
GLOBAL void renderSpheresKernel(unsigned int* offset, SphereInfo* tri, TileRef* idx,
    Uniform* uniform, FrameBuffer* frameBuffer, const float near, const float far,
    const float invnf,const vec2 invmul, const vec2 invHsiz) {
    applySFS<Uniform, FrameBuffer, FragShader...>(offset, tri, idx, uniform, frameBuffer, near, far, 
        invnf, invmul, invHsiz);
}

template <typename Uniform, typename FrameBuffer,
    SphereVertShader<Uniform> VertShader, SphereFragmentShader<Uniform, FrameBuffer>... FragShader>
void renderSpheres(CommandBuffer& buffer, const Span<vec4>& spheres,
                   const Span<Uniform>& uniform, const Span<FrameBuffer>& frameBuffer,
    const uvec2 size, const float near, const float far, const vec2 mul, vec4 scissor) {
    auto cameraSpheres = buffer.allocBuffer<vec4>(spheres.size());
    buffer.launchKernelLinear(calcCameraSpheres<Uniform, VertShader>, spheres.size(), spheres,
                           cameraSpheres, uniform);
    scissor = { fmax(0.5f,scissor.x),fmin(size.x - 0.5f,scissor.y),
        fmax(0.5f,scissor.z),fmin(size.y - 0.5f,scissor.w) };
    const auto hfsize = static_cast<vec2>(size) * 0.5f;
    auto processRes = processSphereInfo(buffer, cameraSpheres, scissor, hfsize, near, far, mul);
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderSpheresKernel<Uniform, FrameBuffer, FragShader...>, processRes.offset,
        processRes.info, processRes.ref, uniform, frameBuffer, near, far, invnf,
        1.0f / mul, 1.0f / hfsize);
}
