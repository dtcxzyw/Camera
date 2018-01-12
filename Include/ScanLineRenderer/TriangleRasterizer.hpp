#pragma once
#include "ScanLine.hpp"
#include <Base/DispatchSystem.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>

template<typename Out>
struct Triangle final {
    vec4 rect;
    vec3 invz;
    mat3 w;
    unsigned int id;
    Out out[3];
};

enum class CullFace {
    Front = 0, Back = 1, None = 2
};

CUDAInline float edgeFunction(vec3 a, vec3 b, vec3 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

template<typename Index,typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) vert, Index index,
    VertexInfo<Out>* info, int mode) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    auto a = vert[idx[0]], b = vert[idx[1]], c = vert[idx[2]];
    float S = edgeFunction(a.pos, b.pos, c.pos);
    if ((S < 0.0) ^ mode) {
        int base=3*atomicInc(cnt, maxv);
        info[base] = a,info[base+1]=b,info[base+2]=c;
    }
}

CUDAInline float max3(float a, float b, float c) {
    return fmax(a, fmax(b, c));
}

CUDAInline float min3(float a, float b, float c) {
    return fmin(a, fmin(b, c));
}

CUDAInline void calcBase(vec3 a, vec3 b, vec3& w) {
    w.x = b.y - a.y, w.y = a.x - b.x;
    w.z = -(a.x*w.x + a.y * w.y);
}

CUDAInline vec3 toRaster(vec3 p, float hx,float hy) {
    auto invz = 1.0f / p.z;
    return { (1.0f+p.x*invz)*hx,(1.0f-p.y*invz)*hy,p.z };
}

template<typename Out>
CALLABLE void processTriangles(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) vert,Triangle<Out>* info, unsigned int* triID,
    float fx,float fy,float hx,float hy) {
    auto id = getID();
    if (id >= size)return;
    auto base = id * 3;
    uvec3 idx{base,base+1,base+2};
    auto a = toRaster(vert[idx[0]].pos,hx,hy),
        b = toRaster(vert[idx[1]].pos,hx,hy), 
        c = toRaster(vert[idx[2]].pos,hx,hy);
    vec4 rect= { fmax(0.0f,min3(a.x,b.x,c.x)),fmin(fx,max3(a.x,b.x,c.x)),
        fmax(0.0f,min3(a.y,b.y,c.y)),fmin(fy,max3(a.y,b.y,c.y)) };
    if (rect.x<rect.y & rect.z<rect.w) {
        Triangle<Out> res;
        res.rect = rect;
        res.invz = {1.0f/ a.z,1.0f/ b.z,1.0f/c.z};
        calcBase(b, c, res.w[0]);
        calcBase(c, a, res.w[1]);
        calcBase(a, b, res.w[2]);
        res.w *= 1.0f / edgeFunction(a, b, c);
        res.id = id;
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

template<typename Out>
CALLABLE void processTrianglesGPU(unsigned int* size, unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) vert, Triangle<Out>* info, unsigned int* triID,
    float fx, float fy, float hx, float hy) {
    constexpr auto block = 1024U;
    if (*size)processTriangles<Out> << <calcSize(*size, block), block >> > (*size, cnt, vert, info, 
        triID, fx, fy, hx, hy);
}

CUDAInline bool testPoint(mat3 w0, vec2 p, vec3& w) {
    w.x = w0[0].x*p.x + w0[0].y*p.y + w0[0].z;
    w.y = w0[1].x*p.x + w0[1].y*p.y + w0[1].z;
    w.z = w0[2].x*p.x + w0[2].y*p.y + w0[2].z;
    return w.x >= 0.0f&w.y >= 0.0f&w.z >= 0.0f;
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDAInline void drawPoint(Triangle<Out> tri, ivec2 uv, vec2 p, Uniform uni, 
        FrameBuffer& frameBuffer,float near,float invnf) {
    vec3 w;
    bool flag = testPoint(tri.w, p, w);
    auto z = 1.0f / dot(tri.invz, w);
    if (flag) {
        w *= z;
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        auto nz = (z - near)*invnf;//convert z to [0,1]
        fs(tri.id,uv, nz, fo, uni, frameBuffer);
    }
}

//1,2,4,8,16,32
template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawMicroT(ReadOnlyCache(Triangle<Out>) info,
        ReadOnlyCache(unsigned int) idx,
        ReadOnlyCache(Uniform) uniform, FrameBuffer* frameBuffer,float near,float invnf) {
    auto tri = info[idx[blockIdx.x]];
    ivec2 uv{ tri.rect.x + threadIdx.x,tri.rect.z + threadIdx.y };
    vec2 p{ uv.x+0.5f,uv.y+0.5f };
    if(p.x<=tri.rect.y & p.y<=tri.rect.w)
        drawPoint<Out, Uniform, FrameBuffer, fs>(tri, uv, p, *uniform, *frameBuffer,near,invnf);
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
    CALLABLE void renderTrianglesGPU(unsigned int* cnt, Triangle<Out>* tri, 
        unsigned int* idx, Uniform* uniform, FrameBuffer* frameBuffer,unsigned int size,
       float near,float invnf) {
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
            drawMicroT<Out, Uniform, FrameBuffer,ds> << <grid, block >> > (tri, idx + size * i,
                uniform, frameBuffer,near,invnf);
        }

    if (cnt[7]) {
        auto bsiz = 1U << 5;
        dim3 grid(cnt[7]);
        dim3 block(bsiz, bsiz);
        drawMicroT<Out, Uniform, FrameBuffer, ds> << <grid, block >> > (tri, idx + size * 7,
            uniform, frameBuffer,near,invnf);
    }

    cudaDeviceSynchronize();

    for (auto i = 0; i < 6; ++i)
        if (cnt[i]) {
            auto bsiz = 1U << i;
            dim3 grid(cnt[i]);
            dim3 block(bsiz, bsiz);
            drawMicroT<Out, Uniform, FrameBuffer, fs> << <grid, block >> > (tri, idx + size * i,
                uniform, frameBuffer,near,invnf);
        }

    if (cnt[7]) {
        auto bsiz = 1U << 5;
        dim3 grid(cnt[7]);
        dim3 block(bsiz, bsiz);
        drawMicroT<Out, Uniform, FrameBuffer, fs> << <grid, block >> > (tri, idx + size * 7,
            uniform, frameBuffer,near,invnf);
    }
}

enum class ClipZMode {
    Near=0,Far=1
};

template<ClipZMode mode>
CUDAInline int compareZ(float z, float base) { return 0; }

template<>
CUDAInline int compareZ <ClipZMode::Near>(float z, float base) { return z < base; }

template<>
CUDAInline int compareZ<ClipZMode::Far>(float z, float base) { return z > base; }

template<typename Out,ClipZMode mode>
CALLABLE void sortTriangles(unsigned int size,ReadOnlyCache(VertexInfo<Out>) vert,
    unsigned int* cnt, unsigned int* clip, float z) {
    auto id = getID();
    if (id >= size)return;
    auto base=id*3;
    uvec3 idx{ base,base + 1,base + 2 };
    int type = 0;
    for (int i = 0; i < 3; ++i)
        type +=compareZ<mode>(vert[idx[i]].pos.z,z);
    if (type < 3) {
        if (compareZ<mode>(vert[idx[1]].pos.z, vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
        if (compareZ<mode>(vert[idx[2]].pos.z, vert[idx[1]].pos.z))cudaSwap(idx[2], idx[1]);
        if (compareZ<mode>(vert[idx[1]].pos.z, vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
        auto wpos=atomicInc(cnt+type,maxv);
        clip[wpos + type * size] =id;
    }
}

template<typename Out>
CALLABLE void clipVertT0(unsigned int size,unsigned int* cnt ,
    ReadOnlyCache(VertexInfo<Out>) in, ReadOnlyCache(unsigned int) clip,
    VertexInfo<Out>* out) {
    auto id = getID();
    if (id >= size)return;
    auto off = clip[id]*3;
    uvec3 idx{ off,off + 1,off + 2 };
    auto base = atomicInc(cnt,maxv)*3;
    for (int i = 0; i < 3; ++i)
        out[base + i] = in[idx[i]];
}

template<typename Out>
CALLABLE void clipVertT1(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) in, ReadOnlyCache(unsigned int) clip,
    VertexInfo<Out>* out,float z) {
    auto id = getID();
    if (id >= size)return;
    auto off = clip[id]*3;
    uvec3 idx{ off,off + 1,off + 2 };
    auto a = in[idx[0]], b = in[idx[1]], c = in[idx[2]];
    auto d = lerpZ(a, b, z), e = lerpZ(a, c, z);
    auto base1 = atomicInc(cnt, maxv) * 3;
    out[base1] = b, out[base1 + 1] = c, out[base1 + 2] = d;
    auto base2= atomicInc(cnt, maxv) * 3;
    out[base2] = d, out[base2 + 1] = e, out[base2 + 2] = c;
}

template<typename Out>
CALLABLE void clipVertT2(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) in, ReadOnlyCache(unsigned int) clip,
    VertexInfo<Out>* out,float z) {
    auto id = getID();
    if (id >= size)return;
    auto off = clip[id]*3;
    uvec3 idx{ off,off + 1,off + 2 };
    auto a = in[idx[0]], b = in[idx[1]], c = in[idx[2]];
    auto d = lerpZ(a, c, z), e = lerpZ(b, c, z);
    auto base = atomicInc(cnt, maxv) * 3;
    out[base] = c, out[base + 1] = d, out[base + 2] = e;
}

template<typename Out>
CALLABLE void clipVertTGPU(unsigned int size,unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) in,ReadOnlyCache(unsigned int) clip,
    VertexInfo<Out>* out,float z) {
    constexpr auto block = 1024U;
    if (cnt[0])clipVertT0<Out> << <calcSize(cnt[0],block), block >> > (cnt[0], cnt+3,in, clip, out);
    if (cnt[1])clipVertT1<Out> << <calcSize(cnt[1],block), block >> > (cnt[1], cnt+3,in, clip+size, out,z);
    if (cnt[2])clipVertT2<Out> << <calcSize(cnt[2],block), block >> > (cnt[2], cnt+3,in, clip+size*2, out,z);
}

template<typename Out, ClipZMode mode>
CALLABLE void sortTrianglesGPU(unsigned int* size, ReadOnlyCache(VertexInfo<Out>) vert,
    unsigned int* cnt, unsigned int* clip, float z) {
    constexpr auto block = 1024U;
    if (*size)sortTriangles<Out, mode> << <calcSize(*size, block), block >> > (*size, vert, cnt, clip, z);
}

template<typename Out,ClipZMode mode>
auto clipVertT(CommandBuffer& buffer,const DataPtr<VertexInfo<Out>>& vert,
    LaunchSize size,float z,size_t tsize) {
    auto outVert=buffer.allocBuffer<VertexInfo<Out>>(tsize*3);
    auto clip = buffer.allocBuffer<unsigned int>(tsize*3);
    auto cnt = buffer.allocBuffer<unsigned int>(4);//c0 c1 c2 triNum
    buffer.memset(cnt);
    buffer.callKernel(sortTrianglesGPU<Out, mode>, size, vert, cnt, clip, z);
    buffer.callKernel(clipVertTGPU<Out>,tsize,cnt,vert,clip,outVert,z);
    return std::make_pair(LaunchSize(cnt,3),outVert);
}
