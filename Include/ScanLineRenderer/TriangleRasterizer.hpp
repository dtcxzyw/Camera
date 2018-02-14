#pragma once
#include "ScanLine.hpp"
#include <Base/DispatchSystem.hpp>
#include <Base/CompileBegin.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>
#include <cuda_device_runtime_api.h>
#include <Base/CompileEnd.hpp>

inline auto calcBufferSize(const unsigned int history, const unsigned int base, const unsigned int maxv) {
    return base + std::min(history + (history >> 3), maxv);
}

template <typename Out>
struct TriangleVert final {
    unsigned int id;
    VertexInfo<Out> vert[3];
};

CUDAINLINE float edgeFunction(const vec3 a, const vec3 b, const vec3 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

CUDAINLINE float max3(const float a, const float b, const float c) {
    return fmax(a, fmax(b, c));
}

CUDAINLINE float min3(const float a, const float b, const float c) {
    return fmin(a, fmin(b, c));
}

CUDAINLINE vec3 calcBase(const vec3 a, const vec3 b) {
    vec3 w;
    w.x = b.y - a.y, w.y = a.x - b.x;
    w.z = -(a.x * w.x + a.y * w.y);
    return w;
}

template <typename Out>
struct Triangle final {
    vec3 invz;
    mat3 w;
    unsigned int id;
    Out out[3];
};

struct TriangleRef final {
    unsigned int id, size;
    vec4 rect;
};

enum class CullFace {
    Front = 0,
    Back = 1,
    None = 2
};

template <typename Out>
struct TriangleProcessingArgs final {
    unsigned int* cnt;
    Triangle<Out>* info;
    TriangleRef* out;
    vec2 fsiz;
    vec2 hsiz;
    int mode;

    TriangleProcessingArgs(unsigned int* iCnt,
                           Triangle<Out>* iInfo, TriangleRef* iOut, const vec2 iFsiz, const vec2 iHsiz, const int iMode)
        : cnt(iCnt), info(iInfo), out(iOut), fsiz(iFsiz), hsiz(iHsiz), mode(iMode) {}
};

CUDAINLINE int calcTileSize(const vec4 rect) {
    const auto tsize = fmax(fmax(rect.y - rect.x, rect.w - rect.z)+0.5f,1.5f);
    return fmin(log2f(tsize),5.5f);
}

template <typename Out>
CUDAINLINE void calcTriangleInfo(TriangleVert<Out> tri, const TriangleProcessingArgs<Out>& args) {
    const auto a = toRaster(tri.vert[0].pos, args.hsiz),
               b = toRaster(tri.vert[1].pos, args.hsiz),
               c = toRaster(tri.vert[2].pos, args.hsiz);
    const vec4 rect = {
        fmax(0.0f, min3(a.x, b.x, c.x)), fmin(args.fsiz.x, max3(a.x, b.x, c.x)),
        fmax(0.0f, min3(a.y, b.y, c.y)), fmin(args.fsiz.y, max3(a.y, b.y, c.y))
    };
    const auto area = edgeFunction(a, b, c);
    if (static_cast<bool>((area < 0.0f) ^ args.mode) & rect.x < rect.y & rect.z < rect.w) {
        Triangle<Out> res;
        res.invz = {a.z, b.z, c.z};
        res.w[0] = calcBase(b, c);
        res.w[1] = calcBase(c, a);
        res.w[2] = calcBase(a, b);
        res.w *= 1.0f / area;
        res.id = tri.id;
        res.out[0] = tri.vert[0].out * res.invz.x;
        res.out[1] = tri.vert[1].out * res.invz.y;
        res.out[2] = tri.vert[2].out * res.invz.z;
        TriangleRef ref;
        ref.size = calcTileSize(rect);
        atomicInc(args.cnt + ref.size, maxv);
        ref.id = atomicInc(args.cnt + 6, maxv);
        ref.rect = rect;
        args.out[ref.id] = ref;
        args.info[ref.id] = res;
    }
}

using CompareZ = bool(*)(float, float);
CUDAINLINE bool compareZNear(const float z, const float base) {
    return z < base;
}

CUDAINLINE bool compareZFar(const float z, const float base) {
    return z > base;
}

template <typename Out, CompareZ Func>
CUDAINLINE int calcTriangleType(TriangleVert<Out> tri, const float z) {
    auto type = 0;
    for (auto i = 0; i < 3; ++i)
        type += Func(tri.vert[i].pos.z, z);
    return type;
}

template <typename Out, CompareZ Func>
CUDAINLINE uvec3 sortIndex(TriangleVert<Out> tri, uvec3 idx) {
    if (Func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    if (Func(tri.vert[idx[2]].pos.z, tri.vert[idx[1]].pos.z))cudaSwap(idx[2], idx[1]);
    if (Func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    return idx;
}

template <typename Out, CompareZ Func, typename Callable>
CUDAINLINE void clipVertT1(TriangleVert<Out> tri, const float z, Callable emit) {
    uvec3 idx = sortIndex<Out, Func>(tri, uvec3{0, 1, 2});
    auto a = tri.vert[idx[0]], b = tri.vert[idx[1]], c = tri.vert[idx[2]];
    auto d = lerpZ(b, a, z), e = lerpZ(c, a, z);
    TriangleVert<Out> out;
    out.id = tri.id;
    if ((idx[1] + 1) % 3 == idx[2])
        out.vert[0] = b, out.vert[1] = c, out.vert[2] = d;
    else
        out.vert[0] = b, out.vert[1] = d, out.vert[2] = c;

    emit(out);

    if ((idx[0] + 1) % 3 == idx[2])
        out.vert[0] = d, out.vert[1] = e, out.vert[2] = c;
    else
        out.vert[0] = e, out.vert[1] = d, out.vert[2] = c;

    emit(out);
}

template <typename Out, CompareZ Func, typename Callable>
CUDAINLINE void clipVertT2(TriangleVert<Out> tri, const float z, Callable emit) {
    uvec3 idx = sortIndex<Out, Func>(tri, uvec3{0, 1, 2});
    auto a = tri.vert[idx[0]], b = tri.vert[idx[1]], c = tri.vert[idx[2]];
    auto d = lerpZ(a, c, z), e = lerpZ(b, c, z);
    TriangleVert<Out> out;
    out.id = tri.id;
    if ((idx[2] + 1) % 3 == idx[0])
        out.vert[0] = d, out.vert[1] = e, out.vert[2] = c;
    else
        out.vert[0] = e, out.vert[1] = d, out.vert[2] = c;

    emit(out);
}

template <typename Out, CompareZ Func, typename Callable>
CUDAINLINE void clipTriangle(TriangleVert<Out> tri, const float z, Callable emit) {
    const auto type = calcTriangleType<Out, Func>(tri, z);
    switch (type) {
    case 0: emit(tri);
        break;
    case 1: clipVertT1<Out, Func, Callable>(tri, z, emit);
        break;
    case 2: clipVertT2<Out, Func, Callable>(tri, z, emit);
        break;
    default: ;
    }
}

template <typename Uniform>
using TCSF = bool(*)(unsigned int idx, vec3& pa, vec3& pb, vec3& pc,const Uniform& uniform);

template <typename Index, typename Out, typename Uniform, TCSF<Uniform> cs>
CALLABLE void processTriangles(const unsigned int size,READONLY(VertexInfo<Out>) vert,
                               Index index,READONLY(Uniform) uniform, const float near, const float far,
                               TriangleProcessingArgs<Out> args) {
    const auto id = getID();
    if (id >= size)return;
    const auto idx = index[id];
    TriangleVert<Out> tri;
    tri.id = id, tri.vert[0] = vert[idx[0]], tri.vert[1] = vert[idx[1]], tri.vert[2] = vert[idx[2]];
    if (cs(id, tri.vert[0].pos, tri.vert[1].pos, tri.vert[2].pos, *uniform)) {
        const auto emitF = [&args](TriangleVert<Out> t) {
            calcTriangleInfo<Out>(t, args);
        };
        const auto emitN = [far,emitF](TriangleVert<Out> t) {
            clipTriangle<Out, compareZFar, decltype(emitF)>(t, far, emitF);
        };
        clipTriangle<Out, compareZNear, decltype(emitN)>(tri, near, emitN);
    }
}

std::pair<MemoryRef<unsigned int>, MemoryRef<TriangleRef>> sortTriangles(CommandBuffer& buffer,
                                                                         const MemoryRef<unsigned int>& cnt,
                                                                         const MemoryRef<TriangleRef>& ref);

CUDAINLINE bool calcWeight(const mat4 w0, const vec2 p, const vec3 invz,
    const float near,const float invnf,vec4& w) {
    w.x = w0[0].x * p.x + w0[0].y * p.y + w0[0].z;
    w.y = w0[1].x * p.x + w0[1].y * p.y + w0[1].z;
    w.z = w0[2].x * p.x + w0[2].y * p.y + w0[2].z;
    const bool flag = w.x >= 0.0f & w.y >= 0.0f & w.z >= 0.0f;
    const auto z = 1.0f / (invz.x*w.x+invz.y*w.y+invz.z*w.z);
    w.x *= z, w.y *= z, w.z *= z;
    w.w = (z - near)*invnf;
    return flag;
}

//2,4,8,16,32
template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer> fs>
CALLABLE void drawMicroT(READONLY(Triangle<Out>) info,READONLY(TriangleRef) idx,
                         READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
                         const float near, const float invnf) {
    const auto ref = idx[blockIdx.x];
    const auto tri = info[ref.id];
    const auto offX = threadIdx.z >> 1U, offY = threadIdx.z & 1U;
    const ivec2 uv{ref.rect.x + (threadIdx.x<<1)+offX, ref.rect.z + (threadIdx.y<<1)+offY};
    const vec2 p{uv.x + 0.5f, uv.y + 0.5f};
    vec4 w;
    const auto flag = calcWeight(tri.w,p,tri.invz,near,invnf,w);
    if (p.x <= ref.rect.y & p.y <= ref.rect.w & flag) {
        const auto fout = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        fs(tri.id,uv,w.w,fout,*uniform,*frameBuffer);
    }
}

template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer> first, FSF<Out, Uniform, FrameBuffer>... then>
CUDAINLINE void applyTFS(unsigned int* offset, Triangle<Out>* tri, TriangleRef* idx, Uniform* uniform,
                         FrameBuffer* frameBuffer, float near, float invnf) {
    for (auto i = 0; i < 5; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            const auto bsiz = 1U << i;
            dim3 grid(size);
            dim3 block(bsiz, bsiz,4);
            drawMicroT<Out, Uniform, FrameBuffer, first> << <grid, block >> >(tri, idx + offset[i],
                                                                              uniform, frameBuffer, near, invnf);
        }
    }

    cudaDeviceSynchronize();
    applyTFS<Out, Uniform, FrameBuffer, then...>(offset, tri, idx, uniform, frameBuffer, near, invnf);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applyTFS(unsigned int*, Triangle<Out>*, TriangleRef*, Uniform*, FrameBuffer*,
                         float, float) {}

template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer>... fs>
CALLABLE void renderTrianglesGPU(unsigned int* offset, Triangle<Out>* tri, TriangleRef* idx,
                                 Uniform* uniform, FrameBuffer* frameBuffer, const float near, const float invnf) {
    applyTFS<Out, Uniform, FrameBuffer, fs...>(offset, tri, idx, uniform, frameBuffer, near, invnf);
}

struct TriangleRenderingHistory final : Uncopyable {
    std::atomic_uint triNum;
    uint64 baseSize;
    bool enableSelfAdaptiveAllocation;

    void reset(const unsigned int size, const unsigned int base = 2048U, const bool SAA = false) {
        triNum = size;
        baseSize = base;
        enableSelfAdaptiveAllocation = SAA;
    }
};

template <typename Index, typename Out, typename Uniform, typename FrameBuffer,
          TCSF<Uniform> cs, FSF<Out, Uniform, FrameBuffer>... fs>
void renderTriangles(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
                     Index index, const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer,
                     const uvec2 size, float near, float far, TriangleRenderingHistory& history,
                     CullFace mode = CullFace::Back) {
    //pass 1:process triangles
    auto psiz = calcBufferSize(history.triNum, history.baseSize, index.size());
    auto cnt = buffer.allocBuffer<unsigned int>(7);
    buffer.memset(cnt);
    auto info = buffer.allocBuffer<Triangle<Out>>(psiz);
    auto idx = buffer.allocBuffer<TriangleRef>(psiz);
    vec2 fsize = size - uvec2{1, 1};
    auto hfsize = static_cast<vec2>(size) * 0.5f;
    buffer.runKernelLinear(processTriangles<Index, Out, Uniform, cs>, index.size(), vert.get(),
                           index, uniform.get(), near, far,
                           buffer.makeLazyConstructor<TriangleProcessingArgs<Out>>(cnt, info, idx, fsize, hfsize,
                                                                                   static_cast<int>(mode)));
    if (history.enableSelfAdaptiveAllocation) {
        LaunchSize triNumData(cnt, 6);
        triNumData.download(history.triNum, buffer);
    }

    //pass 2:sort triangles
    auto sortedTri = sortTriangles(buffer, cnt, idx);
    cnt.earlyRelease();
    idx.earlyRelease();

    //pass 3:render triangles
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderTrianglesGPU<Out, Uniform, FrameBuffer, fs...>, sortedTri.first, info,
                      sortedTri.second, uniform.get(), frameBuffer.get(), near, invnf);
}
