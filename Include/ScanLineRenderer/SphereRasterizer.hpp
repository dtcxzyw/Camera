#pragma once
#include <ScanLineRenderer/Shared.hpp>

//in camera pos
template <typename Uniform, typename FrameBuffer>
using FSFS = void(*)(unsigned int id, ivec2 uv, float z, vec3 pos,vec3 dir,float r,bool inSphere,
    vec2 pdd,const Uniform& uniform, FrameBuffer& frameBuffer);

/*
 *Utility Functions
 *x=sin theta cos phi
 *y=cos theta
 *z=sin theta sin phi
 *u=phi/2pi
 *v=theta/pi
 */
CUDAINLINE vec2 calcSphereTexCoord(const vec3 normalizedWorldDir) {
    const auto theta = std::acos(normalizedWorldDir.y);
    const auto phi = std::atan2(normalizedWorldDir.z,normalizedWorldDir.x);
    return {phi*one_over_two_pi<float>(),theta*one_over_pi<float>()};
}

CUDAINLINE vec3 calcSphereNormal(const vec3 normalizedWorldDir,const bool inSphere) {
    return inSphere ? -normalizedWorldDir : normalizedWorldDir;
}

CUDAINLINE vec3 calcSphereBiTangent(const vec3 normal) {
    const auto left = cross({ 0.0f,1.0f,0.0f },normal);
    return cross(left,normal);
}

CUDAINLINE vec3 calcSphereTangent(const vec3 normal,const vec3 biTangent) {
    return cross(biTangent,normal);
}

//ddx+ddy
CUDAINLINE vec4 calcSphereTdd(const vec3 dir,const vec2 pdd) {
    const auto ddxdu = -two_pi<float>()*dir.z;
    const auto duddx = 1.0f / ddxdu;
    const auto ddxdv = pi<float>()*dir.y*std::cos(std::atan2(dir.z, dir.x));
    const auto dvddx = 1.0f / ddxdv;
    return {pdd.x*duddx,pdd.x*dvddx,pdd.y*duddx,pdd.y*dvddx};
}

struct SphereInfo final {
    unsigned int id;
    vec4 info;
    float c;
};

struct SphereProcessResult final {
    MemoryRef<unsigned int> offset;
    MemoryRef<SphereInfo> info;
    MemoryRef<TileRef> ref;
    SphereProcessResult(const MemoryRef<unsigned int>& off,
        const MemoryRef<SphereInfo>& sphereInfo, const MemoryRef<TileRef>& sphereRef)
        :offset(off), info(sphereInfo), ref(sphereRef){}
};

SphereProcessResult processSphereInfo(
    CommandBuffer& buffer, const DataPtr<vec4>& spheres, vec2 fsiz, vec2 hsiz,
    float near, float far, vec2 mul);

/*
 * pos=td
 * NDC.x=pos.x*mul.x/pos.z
 * kx=pos.x/pos.z=NDC.x/mul.x
 * NDC.y=pos.y*mul.y/pos.z
 * ky=pos.y/pos.z=NDC.y/mul.y
 * NDC.z=-pos.z
 * t=pos.z [-far,-near]
 * d=(kx,ky,1)
 * solve at^2+bt+c=0
 */

CUDAINLINE vec2 toNDC(const vec2 p,const float ihx,const float ihy) {
    return {p.x*ihx-1.0f,1.0f-p.y*ihy};
}

//2,4,8,16,32
template <typename Uniform, typename FrameBuffer,
    FSFS<Uniform, FrameBuffer> fs>
    GLOBAL void drawMicroS(READONLY(SphereInfo) info, READONLY(TileRef) idx,
        READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
        const float near,const float far,const float invnf,const float imx,const float imy,const float ihx,const float ihy) {
    const auto ref = idx[blockIdx.x];
    const auto offX = threadIdx.x >> 1U, offY = threadIdx.x & 1U;
    const ivec2 uv{ ref.rect.x + (threadIdx.y << 1) + offX, ref.rect.z + (threadIdx.z << 1) + offY };
    const vec2 p{ uv.x + 0.5f, uv.y + 0.5f };
    if(p.x>ref.rect.y | p.y>ref.rect.w)return;
    const auto NDC = toNDC(p, ihx, ihy);
    const vec3 d = { NDC.x*imx,NDC.y*imy,1.0f };
    const auto sphere = info[ref.id];
    const vec3 mid = sphere.info;
    const auto a = length2(d);
    const auto b = -2.0f*dot(d, mid);
    const auto c = sphere.c;
    const auto delta = b * b - 4.0f*a*c;
    if(delta<0.0f)return;
    const auto sqrtDelta = sqrt(delta),inv2a=0.5f/a;
    const auto t1 = (sqrtDelta - b)*inv2a, t2 = (-sqrtDelta - b)*inv2a;
    const auto begin = -far, end = -near;
    float t;
    bool inSphere;
    if (begin <= t1 & t1 <= end)t = t1,inSphere=false;
    else if (begin <= t2 & t2 <= end)t = t2,inSphere=true;
    else return;
    const auto pos = t * d;
    const auto ddx = (shuffleFloat(pos.x, 0b10) - pos.x) * (offX ? -1.0f : 1.0f);
    const auto ddy = (shuffleFloat(pos.x, 0b01) - pos.x) * (offY ? -1.0f : 1.0f);
    const auto nz = (-t - near)*invnf;
    fs(sphere.id, p, nz, pos, pos - mid, sphere.info.w, inSphere, { ddx,ddy }, *uniform, *frameBuffer);
}

template <typename Uniform, typename FrameBuffer,
    FSFS<Uniform, FrameBuffer> first, FSFS<Uniform, FrameBuffer>... then>
    CUDAINLINE void applySFS(unsigned int* offset, SphereInfo* info, TileRef* idx, Uniform* uniform,
        FrameBuffer* frameBuffer, const float near,const float far,const float invnf,
        const vec2 invMul, const vec2 invHsiz) {
#pragma unroll
    for (auto i = 0; i < 5; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            dim3 grid(size);
            const auto bsiz = 1 << (i+1);
            dim3 block(bsiz, bsiz);
            drawMicroS<Uniform, FrameBuffer, first> << <grid, block >> >(info, idx + offset[i],
                uniform, frameBuffer, near,far, invnf,invMul.x,invMul.y,invHsiz.x,invHsiz.y);
        }
    }

    cudaDeviceSynchronize();
    applySFS<Uniform, FrameBuffer, then...>(offset, info, idx, uniform, frameBuffer, near,far, invnf,
        invMul,invHsiz);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applySFS(unsigned int*, SphereInfo*, TileRef*, Uniform*, FrameBuffer*,
    const float,const float,const float,const vec2,const vec2) {}

template <typename Uniform, typename FrameBuffer,
    FSFS<Uniform, FrameBuffer>... fs>
    GLOBAL void renderSpheresGPU(unsigned int* offset, SphereInfo* tri, TileRef* idx,
        Uniform* uniform, FrameBuffer* frameBuffer, const float near,const float far,const float invnf,
        const vec2 invmul,const vec2 invHsiz) {
    applySFS<Uniform, FrameBuffer, fs...>(offset, tri, idx, uniform, frameBuffer, near,far, invnf,
        invmul,invHsiz);
}

template <typename Uniform, typename FrameBuffer,FSFS<Uniform, FrameBuffer>... fs>
void renderSpheres(CommandBuffer& buffer, const DataPtr<vec4>& spheres,
                   const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer,
                   const uvec2 size, const float near, const float far, const vec2 mul) {
    const auto fsize = static_cast<vec2>(size) - vec2{0.5f};
    const auto hfsize = static_cast<vec2>(size) * 0.5f;
    auto processRes = processSphereInfo(buffer,spheres,fsize,hfsize,near,far,mul);
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderSpheresGPU<Uniform, FrameBuffer, fs...>, processRes.offset,
        processRes.info, processRes.ref, uniform.get(), frameBuffer.get(), near,far,invnf,
        1.0f/mul,1.0f/hfsize);
}
