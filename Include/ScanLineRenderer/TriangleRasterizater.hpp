#pragma once
#include "ScanLine.hpp"
#include <math_functions.hpp>
#include <device_atomic_functions.h>

template<typename Out>
struct Triangle final {
    vec3 info;
    int flags;
    vec4 rect;
    vec4 pos[3];
    Out out[3];
};

CUDA float edgeFunction(vec4 a, vec4 b, vec4 c);

CUDA bool calcWeight(vec4 a, vec4 b, vec4 c, vec4 p, vec3 info,int flag,vec3& w);

template<typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt, const vec4* ReadOnly pos
    , const Out* ReadOnly out, const uvec3* ReadOnly index, Triangle<Out>* info) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    auto p0 = pos[idx.x], p1 = pos[idx.y], p2 = pos[idx.z];
    if (edgeFunction(p0, p1, p2) > 0.0f) {
        auto& res = info[atomicInc(cnt, size)];
        res.info = { 1.0f / p0.z,1.0f / p1.z,1.0f / p2.z };
        {
            auto edge0 = p2 - p1, edge1 = p0 - p2, edge2 = p1 - p0;
            res.flags |= ((edge0.y == 0.0f & edge0.x > 0.0f) | edge0.y > 0.0f);
            res.flags |= ((edge1.y == 0.0f & edge1.x > 0.0f) | edge1.y > 0.0f) << 1;
            res.flags |= ((edge2.y == 0.0f & edge2.x > 0.0f) | edge2.y > 0.0f) << 2;
        }
        res.rect = { fmin(p0.x,fmin(p1.x,p2.x)),fmax(p0.x,fmax(p1.x,p2.x)),
            fmin(p0.y,fmin(p1.y,p2.y)),fmax(p0.y,fmax(p1.y,p2.y)) };
        res.pos[0] = p0, res.pos[1] = p1, res.pos[2] = p2;
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
    vec4 p{uv.x,uv.y ,0.0f,0.0f };
    if ((p.x<tri.rect.x)|(p.x>tri.rect.y)|(p.y<tri.rect.z)|(p.y>tri.rect.w))return;
    vec3 w;
    auto p0 = tri.pos[0], p1 = tri.pos[1], p2 = tri.pos[2];
    bool flag = calcWeight(p0, p1, p2, p, tri.info,tri.flags,w);
    p.z = p0.z*w.x + p1.z*w.y + p2.z*w.z;
    if (flag & p.z >= 0.0f & p.z <= 1.0f) {
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        fs(uv,p.z, fo, *uniform, *frameBuffer);
    }
}
