#pragma once
#include "ScanLine.hpp"
#include <math_functions.hpp>
#include <device_atomic_functions.h>

template<typename Out>
struct Triangle final {
    vec4 rect;
    mat3 w;
    vec3 z;
    vec3 invz;
    int flags;
    Out out[3];
};

CUDA inline float edgeFunction(vec3 a, vec3 b, vec3 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

CUDA bool calcWeight(mat3 w0, vec2 p, vec3 info, int flag, vec3& w);

CUDA inline void calcBase(vec3 a, vec3 b, vec3& w) {
    w.x = b.y - a.y, w.y = a.x - b.x;
    w.z = -(a.x*w.x + a.y * w.y);
}

constexpr auto maxv = std::numeric_limits<unsigned int>::max();

template<typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt, const vec4* ReadOnly pos
    , const Out* ReadOnly out, const uvec3* ReadOnly index, Triangle<Out>* info) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    vec3 a = pos[idx.x], b = pos[idx.y], c = pos[idx.z];
    if (edgeFunction(a, b, c) > 0.0f
        & (a.z <= 1.0f | b.z <= 1.0f | c.z <= 1.0f) & (a.z >= 0.0f | b.z >= 0.0f | c.z >= 0.0f)) {
        auto& res = info[atomicInc(cnt, maxv)];
        res.rect = { fmin(a.x,fmin(b.x,c.x)),fmax(a.x,fmax(b.x,c.x)),
            fmin(a.y,fmin(b.y,c.y)),fmax(a.y,fmax(b.y,c.y)) };
        calcBase(b, c, res.w[0]);
        calcBase(c, a, res.w[1]);
        calcBase(a, b, res.w[2]);
        res.z = { a.z,b.z,c.z };
        res.invz = { 1.0f / a.z,1.0f / b.z,1.0f / c.z };
        res.flags = ((c.y == b.y & c.x > b.x) | c.y > b.y) | 
            (((a.y == c.y & a.x > c.x) | a.y > c.y) << 1)|
            (((b.y == a.y & b.x > a.x) | b.y > a.y) << 2);
        res.out[0] = out[idx.x], res.out[1] = out[idx.y], res.out[2] = out[idx.z];
    }
}

template<typename Out>
    CALLABLE void clipTile(unsigned int size, const Triangle<Out>* ReadOnly in,
        unsigned int* cnt, Triangle<Out>* out,vec4 rect) {
    auto id = getID();
    if (id >= size)return;
    auto range = in[id].rect;
    if (range.x <= rect.y&range.y >= rect.x&range.z <= rect.w&range.w >= rect.z)
        out[atomicInc(cnt, maxv)]=in[id];
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawTriangles(const Triangle<Out>* ReadOnly info,
        const Uniform* ReadOnly uniform, FrameBuffer* frameBuffer,ivec2 offset) {
    auto tri = info[blockIdx.x];
    ivec2 uv{offset.x+blockIdx.y*blockDim.x + threadIdx.x,
        offset.y+blockIdx.z*blockDim.y + threadIdx.y };
    vec2 p{ uv.x,uv.y };
    if ((p.x < tri.rect.x) | (p.x > tri.rect.y) | (p.y < tri.rect.z) | (p.y > tri.rect.w))return;
    vec3 w;
    bool flag = calcWeight(tri.w, p, tri.invz, tri.flags, w);
    auto z = dot(tri.z, w);
    if (flag & z >= 0.0f & z <= 1.0f) {
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        fs(uv, z, fo, *uniform, *frameBuffer);
    }
}
