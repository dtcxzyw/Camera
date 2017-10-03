#pragma once
#include "ScanLine.hpp"
#include <math_functions.hpp>
#include <device_atomic_functions.h>

template<typename Out>
struct Triangle final {
    vec3 info;
    int flags;
    vec4 rect;
    mat3 w;
    vec3 z;
    Out out[3];
};

CUDA float edgeFunction(vec4 a, vec4 b, vec4 c);

CUDA bool calcWeight(mat3 w0, vec2 p, vec3 info, int flag, vec3& w);

CUDA vec3 calcBase(vec2 a, vec2 b);

template<typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt, const vec4* ReadOnly pos
    , const Out* ReadOnly out, const uvec3* ReadOnly index, Triangle<Out>* info) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    auto a = pos[idx.x], b = pos[idx.y], c = pos[idx.z];
    bool flag1 = a.z <= 1.0f | b.z <= 1.0f | c.z <= 1.0f;
    bool flag2 = a.z >= 0.0f | b.z >= 0.0f | c.z >= 0.0f;
    if (edgeFunction(a, b, c) > 0.0f & flag1 & flag2) {
        auto& res = info[atomicInc(cnt, size)];
        res.info = { 1.0f / a.z,1.0f / b.z,1.0f / c.z };
        {
            auto edge0 = c - b, edge1 = a - c, edge2 = b - a;
            res.flags |= ((edge0.y == 0.0f & edge0.x > 0.0f) | edge0.y > 0.0f);
            res.flags |= ((edge1.y == 0.0f & edge1.x > 0.0f) | edge1.y > 0.0f) << 1;
            res.flags |= ((edge2.y == 0.0f & edge2.x > 0.0f) | edge2.y > 0.0f) << 2;
        }
        res.rect = { fmin(a.x,fmin(b.x,c.x)),fmax(a.x,fmax(b.x,c.x)),
            fmin(a.y,fmin(b.y,c.y)),fmax(a.y,fmax(b.y,c.y)) };
        //w.x = edgeFunction(b, c, p), w.y = edgeFunction(c, a, p), w.z = edgeFunction(a, b, p);
        res.w[0] = calcBase(b, c);
        res.w[1] = calcBase(c, a);
        res.w[2] = calcBase(a, b);
        res.z = { a.z,b.z,c.z };
        res.out[0] = out[idx.x], res.out[1] = out[idx.y], res.out[2] = out[idx.z];
    }
}

constexpr auto tileSize = 32U;

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawTriangles(const Triangle<Out>* ReadOnly info,
        const Uniform* ReadOnly uniform, FrameBuffer* frameBuffer) {
    auto tri = info[blockIdx.x];
    ivec2 uv{ blockIdx.y*tileSize + threadIdx.x,blockIdx.z*tileSize + threadIdx.y };
    vec2 p{ uv.x,uv.y };
    if ((p.x < tri.rect.x) | (p.x > tri.rect.y) | (p.y < tri.rect.z) | (p.y > tri.rect.w))return;
    vec3 w;
    bool flag = calcWeight(tri.w, p, tri.info, tri.flags, w);
    auto z = dot(tri.z, w);
    if (flag & z >= 0.0f & z <= 1.0f) {
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        fs(uv, z, fo, *uniform, *frameBuffer);
    }
}
