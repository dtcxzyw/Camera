#pragma once
#include "ScanLine.hpp"
#include <math_functions.h>
#include <device_atomic_functions.h>

template<typename Out>
struct Triangle final {
    vec4 rect;
    vec3 invz;
    mat3 w;
    Out out[3];
};

CUDAInline float edgeFunction(vec3 a, vec3 b, vec3 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

CUDAInline bool testPoint(mat3 w0, vec2 p, vec3& w) {
    w.x = w0[0].x*p.x + w0[0].y*p.y + w0[0].z;
    w.y = w0[1].x*p.x + w0[1].y*p.y + w0[1].z;
    w.z = w0[2].x*p.x + w0[2].y*p.y + w0[2].z;
    return w.x >= 0.0f&w.y >= 0.0f&w.z >= 0.0f;
}

CUDAInline void calcBase(vec3 a, vec3 b, vec3& w) {
    w.x = b.y - a.y, w.y = a.x - b.x;
    w.z = -(a.x*w.x + a.y * w.y);
}

template<typename Index, typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt,
    const VertexInfo<Out>* ReadOnlyCache vert, Index index,
    Triangle<Out>* info, unsigned int* triID, vec2 fsize) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    vec3 a = vert[idx[0]].pos, b = vert[idx[1]].pos, c = vert[idx[2]].pos;
    float S = edgeFunction(a,b,c);
    vec4 rect= { fmax(0.0f,fmin(a.x,fmin(b.x,c.x))),fmin(fsize.x,fmax(a.x,fmax(b.x,c.x))),
        fmax(0.0f,fmin(a.y,fmin(b.y,c.y))),fmin(fsize.y,fmax(a.y,fmax(b.y,c.y))) };
    float minz = fmax(0.0f, fmin(a.z, fmin(b.z, c.z))), maxz = fmin(1.0f, fmax(a.z, fmax(b.z, c.z)));
    if (rect.x<rect.y & rect.z<rect.w & minz<=maxz) {
        Triangle<Out> res;
        res.rect = rect;
        res.invz = {1.0f/ a.z,1.0f/ b.z,1.0f/c.z };
        calcBase(b, c, res.w[0]);
        calcBase(c, a, res.w[1]);
        calcBase(a, b, res.w[2]);
        res.w *= 1.0f / S;
        res.out[0] = vert[idx[0]].out*res.invz.x;
        res.out[1] = vert[idx[1]].out*res.invz.y;
        res.out[2] = vert[idx[2]].out*res.invz.z;
        auto tsize = fmax(res.rect.y - res.rect.x, res.rect.w - res.rect.z);
        auto x=static_cast<int>(ceil(log2f(fmin(tsize + 1.0f, 50.0f))));
        auto tid = atomicInc(cnt + 8, maxv);
        triID[atomicInc(cnt + x, maxv) + x * size] = tid;
        info[tid] = res;
    }
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDAInline void drawPoint(Triangle<Out> tri, ivec2 uv, vec2 p, Uniform uni, FrameBuffer& frameBuffer) {
    vec3 w;
    bool flag = testPoint(tri.w, p, w);
    auto z = 1.0f/dot(tri.invz, w);
    if (flag & z >= 0.0f & z <= 1.0f) {
        w *= z;
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        fs(uv, z, fo, uni, frameBuffer);
    }
}

//1,2,4,8,16,32
template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawMicroT(const Triangle<Out>* ReadOnlyCache info,
        const unsigned int* ReadOnlyCache idx,
        const Uniform* ReadOnlyCache uniform, FrameBuffer* frameBuffer) {
    auto tri = info[idx[blockIdx.x]];
    ivec2 uv{ tri.rect.x + threadIdx.x,tri.rect.z + threadIdx.y };
    vec2 p{ uv.x+0.5f,uv.y+0.5f };
    if(p.x<=tri.rect.y & p.y<=tri.rect.w)
        drawPoint<Out, Uniform, FrameBuffer, fs>(tri, uv, p, *uniform, *frameBuffer);
}

template<typename Out>
CALLABLE void cutTriangles(unsigned int size, unsigned int* cnt,
    Triangle<Out>* tri, unsigned int* idx, unsigned int* out) {
    auto id = getID();
    if (id >= size)return;
    auto info = tri[idx[id]];
    auto rect = info.rect;
    for (int i = rect[0]; i <= rect[1]; i += 32) {
        for (int j = rect[2]; j <= rect[3]; j += 32) {
            auto tid = atomicInc(cnt + 8, maxv);
            out[atomicInc(cnt + 7, maxv)] = tid;
            info.rect[0] = i, info.rect[2] = j;
            tri[tid] = info;
        }
    }
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void renderTrianglesGPU(unsigned int* cnt, Triangle<Out>* tri
        , unsigned int* idx, Uniform* uniform, FrameBuffer* frameBuffer, unsigned int size) {
    if (cnt[6]) {
        constexpr auto block = 128U;
        cutTriangles<Out> << <calcSize(cnt[6], block), block >> > (cnt[6], cnt, tri, idx + size * 6, idx + size * 7);
        cudaDeviceSynchronize();
    }

    for (auto i = 0; i < 6; ++i)
        if (cnt[i]) {
            auto bsiz = 1U << i;
            dim3 grid(cnt[i]);
            dim3 block(bsiz, bsiz);
            drawMicroT<Out, Uniform, FrameBuffer, ds> << <grid, block >> > (tri, idx + size * i,
                uniform, frameBuffer);
        }

    if (cnt[7]) {
        auto bsiz = 1U << 5;
        dim3 grid(cnt[7]);
        dim3 block(bsiz, bsiz);
        drawMicroT<Out, Uniform, FrameBuffer, ds> << <grid, block >> > (tri, idx + size * 7,
            uniform, frameBuffer);
    }

    cudaDeviceSynchronize();

    for (auto i = 0; i < 6; ++i)
        if (cnt[i]) {
            auto bsiz = 1U << i;
            dim3 grid(cnt[i]);
            dim3 block(bsiz, bsiz);
            drawMicroT<Out, Uniform, FrameBuffer, fs> << <grid, block >> > (tri, idx + size * i,
                uniform, frameBuffer);
        }

    if (cnt[7]) {
        auto bsiz = 1U << 5;
        dim3 grid(cnt[7]);
        dim3 block(bsiz, bsiz);
        drawMicroT<Out, Uniform, FrameBuffer, fs> << <grid, block >> > (tri, idx + size * 7,
            uniform, frameBuffer);
    }
}

