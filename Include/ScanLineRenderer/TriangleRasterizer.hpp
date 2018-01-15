#pragma once
#include "ScanLine.hpp"
#include <Base/DispatchSystem.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>

template<typename Out>
struct TriangleVert final {
    unsigned int id;
    VertexInfo<Out> vert[3];
};

enum class CullFace {
    Front = 0, Back = 1, None = 2
};

template<typename T>
CUDAInline float edgeFunction(T a, T b, T c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

CUDAInline vec2 fixPos(vec3 p, float kx, float ky) {
    auto invz = 1.0f / fabs(p.z);
    return { p.x*kx*invz,p.y*ky*invz };
}

template<typename Index,typename Out>
CALLABLE void clipTriangles(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(VertexInfo<Out>) vert, Index index,
    TriangleVert<Out>* out, int mode,float kx,float ky) {
    auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    auto a = vert[idx[0]], b = vert[idx[1]], c = vert[idx[2]];
    auto pa = fixPos(a.pos, kx, ky),
        pb = fixPos(b.pos,kx, ky),
        pc = fixPos(c.pos, kx, ky);
    auto S = edgeFunction(pa,pb,pc);
    if ((S > 0.0f) ^ mode) {
        auto base=atomicInc(cnt, maxv);
        out[base].id = id;
        out[base].vert[0] = a,out[base].vert[1]=b,out[base].vert[2]=c;
    }
}

using CompareZ = bool(*)(float, float);
CUDAInline bool compareZNear(float z, float base) { return z < base; }
CUDAInline bool compareZFar(float z, float base) { return z > base; }

template<typename Out, CompareZ func>
CALLABLE void sortTriangles(unsigned int size, ReadOnlyCache(TriangleVert<Out>) in,
    unsigned int* cnt, unsigned int* clip, float z, unsigned int tsize) {
    auto id = getID();
    if (id >= size)return;
    auto tri = in[id];
    int type = 0;
    for (int i = 0; i < 3; ++i)
        type += func(tri.vert[i].pos.z, z);
    if (type < 3) {
        auto wpos = atomicInc(cnt + type, maxv);
        clip[wpos + type * tsize] = id;
    }
}

template<typename Out>
CALLABLE void clipVertT0(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(TriangleVert<Out>) in, ReadOnlyCache(unsigned int) clip,
    TriangleVert<Out>* out) {
    auto id = getID();
    if (id >= size)return;
    out[atomicInc(cnt, maxv)] = in[clip[id]];
}

template<typename Out, CompareZ func>
CUDAInline uvec3 sortIndex(TriangleVert<Out> tri, uvec3 idx) {
    if (func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    if (func(tri.vert[idx[2]].pos.z, tri.vert[idx[1]].pos.z))cudaSwap(idx[2], idx[1]);
    if (func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    return idx;
}

template<typename Out, CompareZ func>
CALLABLE void clipVertT1(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(TriangleVert<Out>) in, ReadOnlyCache(unsigned int) clip,
    TriangleVert<Out>* out, float z) {
    auto id = getID();
    if (id >= size)return;
    auto tri = in[clip[id]];
    uvec3 idx = sortIndex<Out, func>(tri, uvec3{ 0,1,2 });
    auto a = tri.vert[idx[0]], b = tri.vert[idx[1]], c = tri.vert[idx[2]];
    auto d = lerpZ(b, a, z), e = lerpZ(c, a, z);

    auto base1 = atomicInc(cnt, maxv);
    out[base1].id = tri.id;
    out[base1].vert[0] = b, out[base1].vert[1] = c, out[base1].vert[2] = d;

    auto base2 = atomicInc(cnt, maxv);
    out[base2].id = tri.id;
    out[base2].vert[0] = d, out[base2].vert[1] = e, out[base2].vert[2] = c;

}

template<typename Out, CompareZ func>
CALLABLE void clipVertT2(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(TriangleVert<Out>) in, ReadOnlyCache(unsigned int) clip,
    TriangleVert<Out>* out, float z) {
    auto id = getID();
    if (id >= size)return;
    auto tri = in[clip[id]];
    uvec3 idx = sortIndex<Out, func>(tri, uvec3{ 0,1,2 });
    auto a = tri.vert[idx[0]], b = tri.vert[idx[1]], c = tri.vert[idx[2]];
    auto d = lerpZ(a, c, z), e = lerpZ(b, c, z);
    auto base = atomicInc(cnt, maxv);
    out[base].id = tri.id;
    out[base].vert[0] = d, out[base].vert[1] = e, out[base].vert[2] = c;
}

template<typename Out, CompareZ func>
CALLABLE void clipVertTGPU(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(TriangleVert<Out>) in, ReadOnlyCache(unsigned int) clip,
    TriangleVert<Out>* out, float z) {
    constexpr auto block = 1024U;
    run(clipVertT0<Out>, block, cnt[0], cnt + 3, in, clip, out);
    run(clipVertT1<Out, func>, block, cnt[1], cnt + 3, in, clip + size, out, z);
    run(clipVertT2<Out, func>, block, cnt[2], cnt + 3, in, clip + size * 2, out, z);
}

template<typename Out, CompareZ func>
CALLABLE void sortTrianglesGPU(unsigned int* size, ReadOnlyCache(TriangleVert<Out>) in,
    unsigned int* cnt, unsigned int* clip, float z, unsigned int tsize) {
    constexpr auto block = 1024U;
    run(sortTriangles<Out, func>, block, *size, in, cnt, clip, z, tsize);
}

template<typename Out, CompareZ func>
auto clipVertT(CommandBuffer& buffer, const MemoryRef<TriangleVert<Out>>& vert,
    LaunchSize size, float z, unsigned int tsize) {
    auto outVert = buffer.allocBuffer<TriangleVert<Out>>(tsize);
    auto clip = buffer.allocBuffer<unsigned int>(tsize * 3);
    auto cnt = buffer.allocBuffer<unsigned int>(4);//c0 c1 c2 triNum
    buffer.memset(cnt);
    buffer.callKernel(sortTrianglesGPU<Out, func>, size.get(), vert, cnt, clip, z, tsize);
    buffer.callKernel(clipVertTGPU<Out, func>, tsize, cnt, vert, clip, outVert, z);
    return std::make_pair(LaunchSize(cnt, 3), outVert);
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

template<typename Out>
struct Triangle final {
    vec3 invz;
    mat3 w;
    unsigned int id;
    Out out[3];
};

struct TriangleRef final {
    unsigned int id;
    vec4 rect;
};

template<typename Out>
CALLABLE void processTriangles(unsigned int size, unsigned int* cnt,
    ReadOnlyCache(TriangleVert<Out>) in,Triangle<Out>* info,TriangleRef* out,
    float fx,float fy, float hx, float hy, float kx, float ky,unsigned int tsiz) {
    auto id = getID();
    if (id >= size)return;
    auto tri = in[id];
    auto a = toRaster(tri.vert[0].pos,hx,hy,kx,ky),
        b = toRaster(tri.vert[1].pos,hx,hy,kx,ky),
        c = toRaster(tri.vert[2].pos,hx,hy,kx,ky);
    vec4 rect= { fmax(0.0f,min3(a.x,b.x,c.x)),fmin(fx,max3(a.x,b.x,c.x)),
        fmax(0.0f,min3(a.y,b.y,c.y)),fmin(fy,max3(a.y,b.y,c.y)) };
    if (rect.x<rect.y & rect.z<rect.w) {
        Triangle<Out> res;
        res.invz = {a.z, b.z,c.z};
        calcBase(b, c, res.w[0]);
        calcBase(c, a, res.w[1]);
        calcBase(a, b, res.w[2]);
        res.w *= 1.0f / edgeFunction(a, b, c);
        res.id = tri.id;
        res.out[0] = tri.vert[0].out*res.invz.x;
        res.out[1] = tri.vert[1].out*res.invz.y;
        res.out[2] = tri.vert[2].out*res.invz.z;
        auto tsize = fmax(rect.y - rect.x, rect.w - rect.z);
        auto x=static_cast<int>(ceil(log2f(fmin(tsize + 1.0f, 50.0f))));
        TriangleRef ref;
        ref.id = atomicInc(cnt + 8, maxv);
        ref.rect = rect;
        out[atomicInc(cnt + x, maxv) + x * tsiz] = ref;
        info[ref.id] = res;
    }
}

template<typename Out>
CALLABLE void processTrianglesGPU(unsigned int* size, unsigned int* cnt,
    ReadOnlyCache(TriangleVert<Out>) in, Triangle<Out>* info, TriangleRef* triID,
    float fx, float fy, float hx, float hy,float kx,float ky,unsigned int tsiz) {
    constexpr auto block = 1024U;
    run(processTriangles<Out>,block,*size, cnt, in, info, triID, fx, fy, hx, hy,kx,ky,tsiz);
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
        ReadOnlyCache(TriangleRef) idx,
        ReadOnlyCache(Uniform) uniform, FrameBuffer* frameBuffer,float near,float invnf) {
    auto ref = idx[blockIdx.x];
    auto tri = info[ref.id];
    ivec2 uv{ ref.rect.x + threadIdx.x,ref.rect.z + threadIdx.y };
    vec2 p{ uv.x+0.5f,uv.y+0.5f };
    if(p.x<=ref.rect.y & p.y<=ref.rect.w)
        drawPoint<Out, Uniform, FrameBuffer, fs>(tri, uv, p, *uniform, *frameBuffer,near,invnf);
}

template<typename Out>
CALLABLE void cutTriangles(unsigned int size, unsigned int* cnt,TriangleRef* idx, TriangleRef* out) {
    auto id = getID();
    if (id >= size)return; 
    auto ref = idx[id];
    auto rect = ref.rect;
    for (float i = rect[0]; i <= rect[1]; i += 31.0f) {
        for (float j = rect[2]; j <= rect[3]; j += 31.0f) {
            ref.rect[0] = i, ref.rect[2] = j;
            out[atomicInc(cnt + 7, maxv)] = ref;
        }
    }
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void renderTrianglesGPU(unsigned int* cnt, Triangle<Out>* tri, 
        TriangleRef* idx, Uniform* uniform, FrameBuffer* frameBuffer,unsigned int size,
       float near,float invnf) {
    if (cnt[6]) {
        constexpr auto block = 1024U;
        run(cutTriangles<Out>,block,cnt[6], cnt, idx + size * 6, idx + size * 7);
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

