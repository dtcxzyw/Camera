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

CUDA bool calcWeight(mat3 w0, vec2 p, vec3 invz, int flag, vec3& w);

CUDA inline void calcBase(vec3 a, vec3 b, vec3& w) {
    w.x = b.y - a.y, w.y = a.x - b.x;
    w.z = -(a.x*w.x + a.y * w.y);
}

constexpr auto maxv = std::numeric_limits<unsigned int>::max();

template<typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt,
    const VertexInfo<Out>* ReadOnly vert , const uvec3* ReadOnly index,
    Triangle<Out>* info,vec2 pixel) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    vec3 a = vert[idx.x].pos, b = vert[idx.y].pos, c = vert[idx.z].pos;
    if (edgeFunction(a, b, c) > 0.0f 
        & (vert[idx.x].flag|vert[idx.y].flag|vert[idx.z].flag)==0b111111) {
        auto& res = info[atomicInc(cnt, maxv)];
        res.rect = { fmin(a.x,fmin(b.x,c.x)),fmax(a.x,fmax(b.x,c.x)),
            fmin(a.y,fmin(b.y,c.y)),fmax(a.y,fmax(b.y,c.y)) };
        calcBase(b, c, res.w[0]);
        calcBase(c, a, res.w[1]);
        calcBase(a, b, res.w[2]);
        res.z = { a.z,b.z,c.z };
        res.invz = { 1.0f / a.z,1.0f / b.z,1.0f / c.z };
        res.flags = ((c.y == b.y & c.x > b.x) | c.y > b.y) |
            (((a.y == c.y & a.x > c.x) | a.y > c.y) << 1) |
            (((b.y == a.y & b.x > a.x) | b.y > a.y) << 2);
        res.out[0] = vert[idx.x].out, res.out[1] = vert[idx.y].out, res.out[2] = vert[idx.z].out;
    }
}

template<typename Out>
CALLABLE void clipTile(const Triangle<Out>* ReadOnly in,
    unsigned int* cnt, unsigned int* out, unsigned int len) {
    auto id = threadIdx.x*blockDim.y + threadIdx.y;
    auto range = in[blockIdx.x].rect;
    vec2 begin = { len*threadIdx.x,len*threadIdx.y };
    if ((range.x <= begin.x + len) &(range.y >= begin.x)&(range.z <= begin.y + len)&(range.w >= begin.y))
        out[gridDim.x*id + atomicInc(cnt + id, maxv)] = blockIdx.x;
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawTriangles(const Triangle<Out>* ReadOnly info, const unsigned int* ReadOnly tid,
        const Uniform* ReadOnly uniform, FrameBuffer* frameBuffer
        , unsigned int offsetX, unsigned int offsetY) {
    auto tri = info[tid[blockIdx.x]];
    ivec2 uv{ offsetX + blockIdx.y*blockDim.x + threadIdx.x,
        offsetY + blockIdx.z*blockDim.y + threadIdx.y };
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

constexpr auto tileSize = 16U, clipSize = 2U, range = tileSize*clipSize;

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawTile(const unsigned int* ReadOnly tsiz, const Triangle<Out>* ReadOnly info,
        const unsigned int* ReadOnly tid, const Uniform* ReadOnly uniform, FrameBuffer* frameBuffer,
        unsigned int num) {
    auto id = threadIdx.x*blockDim.y + threadIdx.y;
    if (tsiz[id]) {
        dim3 grid(tsiz[id], clipSize, clipSize);
        dim3 block(tileSize, tileSize);
        drawTriangles<Out, Uniform, FrameBuffer, fs> <<<grid, block >>> (info, 
            tid + num*id, uniform, frameBuffer, threadIdx.x*range, threadIdx.y*range);
    }
}
