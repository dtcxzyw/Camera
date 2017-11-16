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
    Out out[3];
};

class UniqueIndexHelper final {
private:
    const unsigned int off;
public:
    CUDA UniqueIndexHelper(unsigned int x) :off(x*3) {}
    CUDA unsigned int operator[](unsigned int x) {
        return off + x;
    }
};

class UniqueIndex final {
private:
    const unsigned int mSize;
public:
    UniqueIndex(unsigned int size):mSize(size){}
    auto size() const {
        return mSize;
    }
    CUDA UniqueIndexHelper operator[](unsigned int off) const {
        return off;
    }
};

class SharedIndex final{
private:
    const uvec3* ReadOnly const mPtr;
    const unsigned int mSize;
public:
    SharedIndex(const uvec3* ReadOnly idx,unsigned int size) :mPtr(idx),mSize(size) {}
    SharedIndex(DataViewer<uvec3> ibo):mPtr(ibo.begin()),mSize(ibo.size()){}
    auto size() const {
        return mSize;
    }
    CUDA auto operator[](unsigned int off) const {
        return mPtr[off];
    }
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

CUDAInline bool calcWeight(mat3 w0, vec2 p, vec3 invz, vec3& w) {
    bool res = testPoint(w0, p, w);
    w /= dot(invz, w);
    w *= invz;
    return res;
}

CUDAInline void calcBase(vec3 a, vec3 b, vec3& w) {
    w.x = b.y - a.y, w.y = a.x - b.x;
    w.z = -(a.x*w.x + a.y * w.y);
}

constexpr auto maxv = std::numeric_limits<unsigned int>::max();

template<typename Index,typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt,
    const VertexInfo<Out>* ReadOnly vert , Index index,
    Triangle<Out>* info,vec2 fsize) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    vec3 a = vert[idx[0]].pos, b = vert[idx[1]].pos, c = vert[idx[2]].pos;
    Triangle<Out> res;
    res.rect = { fmax(0.0f,fmin(a.x,fmin(b.x,c.x))),fmin(fsize.x,fmax(a.x,fmax(b.x,c.x))),
        fmax(0.0f,fmin(a.y,fmin(b.y,c.y))),fmin(fsize.y,fmax(a.y,fmax(b.y,c.y))) };
    auto tsize = fmax(res.rect.y - res.rect.x, res.rect.w - res.rect.z);
    if ((edgeFunction(a, b, c) > 0.0f) & ((vert[idx[0]].flag|vert[idx[1]].flag|vert[idx[2]].flag)==0b111111)) {
        calcBase(b, c, res.w[0]);
        calcBase(c, a, res.w[1]);
        calcBase(a, b, res.w[2]);
        res.z = { a.z,b.z,c.z };
        res.invz = { 1.0f / a.z,1.0f / b.z,1.0f / c.z };
        res.out[0] = vert[idx[0]].out, res.out[1] = vert[idx[1]].out, res.out[2] = vert[idx[2]].out;
        auto x= static_cast<int>(ceil(log2f(fmin(tsize+1.0f, 50.0f))));
        info[x*size+atomicInc(cnt+x, maxv)] = res;
    }
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDAInline void drawPoint(Triangle<Out> tri, ivec2 uv,vec2 p, Uniform uni,FrameBuffer& frameBuffer) {
    vec3 w;
    bool flag = calcWeight(tri.w, p, tri.invz, w);
    auto z = dot(tri.z, w);
    if (flag & z >= 0.0f & z <= 1.0f) {
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        fs(uv, z, fo, uni, frameBuffer);
    }
}

//1,2,4,8,16,32
template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawMicro(const Triangle<Out>* ReadOnly info,
        const Uniform* ReadOnly uniform, FrameBuffer* frameBuffer) {
    auto tri = info[blockIdx.x];
    ivec2 uv{tri.rect.x+threadIdx.x,tri.rect.z+threadIdx.y};
    vec2 p{ uv.x,uv.y };
    drawPoint<Out, Uniform, FrameBuffer, fs>(tri, uv, p, *uniform, *frameBuffer);
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
    drawPoint<Out, Uniform, FrameBuffer, fs>(tri, uv, p, *uniform, *frameBuffer);
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
