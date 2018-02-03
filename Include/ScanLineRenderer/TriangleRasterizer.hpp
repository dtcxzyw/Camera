#pragma once
#include "ScanLine.hpp"
#include <Base/DispatchSystem.hpp>
#include <Base/CompileBegin.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>
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
    return  w;
}

template <typename Out>
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

enum class CullFace {
    Front = 0,
    Back = 1,
    None = 2
};

template<typename Out>
struct TriangleProcessingArgs final {
    unsigned int* cnt; 
    Triangle<Out>* info;
    TriangleRef* out; 
    vec2 fsiz;
    vec2 hsiz;
    unsigned int tsiz;
    int mode;
    TriangleProcessingArgs(unsigned int* iCnt,
        Triangle<Out>* iInfo, TriangleRef* iOut, const vec2 iFsiz, const vec2 iHsiz, const unsigned int iTsiz,
        const int iMode):cnt(iCnt),info(iInfo),out(iOut),fsiz(iFsiz),hsiz(iHsiz),tsiz(iTsiz),mode(iMode) {}
};

template <typename Out>
CUDAINLINE void calcTriangleInfo(TriangleVert<Out> tri,const TriangleProcessingArgs<Out>& args) {
    const auto a = toRaster(tri.vert[0].pos,args.hsiz),
        b = toRaster(tri.vert[1].pos, args.hsiz),
        c = toRaster(tri.vert[2].pos, args.hsiz);
    const vec4 rect = {
        fmax(0.0f, min3(a.x, b.x, c.x)), fmin(args.fsiz.x, max3(a.x, b.x, c.x)),
        fmax(0.0f, min3(a.y, b.y, c.y)), fmin(args.fsiz.y, max3(a.y, b.y, c.y))
    };
    const auto S = edgeFunction(a, b, c);
    if (static_cast<bool>((S < 0.0f) ^ args.mode) & rect.x < rect.y & rect.z < rect.w) {
        Triangle<Out> res;
        res.invz = { a.z, b.z, c.z };
        res.w[0] = calcBase(b, c);
        res.w[1] = calcBase(c, a);
        res.w[2] = calcBase(a, b);
        res.w *= 1.0f / S;
        res.id = tri.id;
        res.out[0] = tri.vert[0].out * res.invz.x;
        res.out[1] = tri.vert[1].out * res.invz.y;
        res.out[2] = tri.vert[2].out * res.invz.z;
        const auto tsize = fmax(rect.y - rect.x, rect.w - rect.z);
        const auto x = static_cast<int>(ceil(log2f(fmin(tsize + 1.0f, 50.0f))));
        TriangleRef ref;
        ref.id = atomicInc(args.cnt + 8, maxv);
        ref.rect = rect;
        args.out[atomicInc(args.cnt + x, maxv) + x * args.tsiz] = ref;
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
    return  type;
}

template <typename Out, CompareZ Func>
CUDAINLINE uvec3 sortIndex(TriangleVert<Out> tri, uvec3 idx) {
    if (Func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    if (Func(tri.vert[idx[2]].pos.z, tri.vert[idx[1]].pos.z))cudaSwap(idx[2], idx[1]);
    if (Func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    return idx;
}

template <typename Out, CompareZ Func,typename Callable>
CUDANOINLINE void clipVertT1(TriangleVert<Out> tri,const float z,Callable emit) {
    uvec3 idx = sortIndex<Out,Func>(tri, uvec3{ 0, 1, 2 });
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

template <typename Out, CompareZ Func,typename Callable>
CUDANOINLINE void clipVertT2(TriangleVert<Out> tri,const float z,Callable emit) {
    uvec3 idx = sortIndex<Out, Func>(tri, uvec3{ 0, 1, 2 });
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

template <typename Out,CompareZ Func,typename Callable>
CUDAINLINE void clipTriangle(TriangleVert<Out> tri, const float z,Callable emit) {
    const auto type = calcTriangleType<Out, Func>(tri,z);
    switch (type) {
    case 0:emit(tri); break;
    case 1:clipVertT1<Out, Func, Callable>(tri, z, emit); break;
    case 2:clipVertT2<Out, Func, Callable>(tri, z, emit); break;
    default:;
    }
}

template <typename Uniform>
using TCSF = bool(*)(unsigned int idx, vec3& pa, vec3& pb, vec3& pc, Uniform uniform);

template <typename Index, typename Out, typename Uniform, TCSF<Uniform> cs>
CALLABLE void processTriangles(const unsigned int size,READONLY(VertexInfo<Out>) vert,
    Index index,READONLY(Uniform) uniform,const float near, const float far,
    TriangleProcessingArgs<Out> args) {
    const auto id = getID();
    if (id >= size)return;
    const auto idx = index[id];
    TriangleVert<Out> tri;
    tri.id = id, tri.vert[0] = vert[idx[0]], tri.vert[1] = vert[idx[1]], tri.vert[2] = vert[idx[2]];
    if (cs(id, tri.vert[0].pos, tri.vert[1].pos, tri.vert[2].pos, *uniform)) {
        const auto emitF=[&args](TriangleVert<Out> t) {
            calcTriangleInfo<Out>(t,args);
        };
        const auto emitN= [far,emitF](TriangleVert<Out> t) {
            clipTriangle<Out, compareZFar, decltype(emitF)>(t, far, emitF);
        };
        clipTriangle<Out,compareZNear,decltype(emitN)>(tri, near,emitN);
    }
}

CUDAINLINE bool testPoint(mat3 w0, const vec2 p, vec3& w) {
    w.x = w0[0].x * p.x + w0[0].y * p.y + w0[0].z;
    w.y = w0[1].x * p.x + w0[1].y * p.y + w0[1].z;
    w.z = w0[2].x * p.x + w0[2].y * p.y + w0[2].z;
    return w.x >= 0.0f & w.y >= 0.0f & w.z >= 0.0f;
}

template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer> fs>
CUDAINLINE void drawPoint(Triangle<Out> tri, ivec2 uv, const vec2 p, Uniform uni,
                          FrameBuffer& frameBuffer, const float near, const float invnf) {
    vec3 w;
    if (testPoint(tri.w, p, w)) {
        auto z = 1.0f / dot(tri.invz, w);
        w *= z;
        auto fo = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        auto nz = (z - near) * invnf; //convert z to [0,1]
        fs(tri.id, uv, nz, fo, uni, frameBuffer);
    }
}

//1,2,4,8,16,32
template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer> fs>
CALLABLE void drawMicroT(READONLY(Triangle<Out>) info,READONLY(TriangleRef) idx,
                         READONLY(Uniform) uniform, FrameBuffer* frameBuffer, float near, float invnf) {
    const auto ref = idx[blockIdx.x];
    auto tri = info[ref.id];
    ivec2 uv{ref.rect.x + threadIdx.x, ref.rect.z + threadIdx.y};
    vec2 p{uv.x + 0.5f, uv.y + 0.5f};
    if (p.x <= ref.rect.y & p.y <= ref.rect.w)
        drawPoint<Out, Uniform, FrameBuffer, fs>(tri, uv, p, *uniform, *frameBuffer, near, invnf);
}

template <typename Out>
CALLABLE void cutTriangles(const unsigned int size, unsigned int* cnt, TriangleRef* idx, TriangleRef* out) {
    const auto id = getID();
    if (id >= size)return;
    auto ref = idx[id];
    auto rect = ref.rect;
    for (float i = rect[0]; i <= rect[1]; i += 32.0f) {
        for (float j = rect[2]; j <= rect[3]; j += 32.0f) {
            ref.rect[0] = i, ref.rect[2] = j;
            out[atomicInc(cnt + 7, maxv)] = ref;
        }
    }
}

template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer> first, FSF<Out, Uniform, FrameBuffer>... then>
CUDAINLINE void applyTFS(unsigned int* cnt, Triangle<Out>* tri,
                         TriangleRef* idx, Uniform* uniform, FrameBuffer* frameBuffer, unsigned int size,
                         float near, float invnf) {
    for (auto i = 0; i < 6; ++i)
        if (cnt[i]) {
            const auto bsiz = 1U << i;
            dim3 grid(cnt[i]);
            dim3 block(bsiz, bsiz);
            drawMicroT<Out, Uniform, FrameBuffer, first> << <grid, block >> >(tri, idx + size * i,
                                                                              uniform, frameBuffer, near, invnf);
        }

    if (cnt[7]) {
        constexpr auto bsiz = 1U << 5;
        dim3 grid(cnt[7]);
        dim3 block(bsiz, bsiz);
        drawMicroT<Out, Uniform, FrameBuffer, first> << <grid, block >> >(tri, idx + size * 7,
                                                                          uniform, frameBuffer, near, invnf);
    }

    cudaDeviceSynchronize();
    applyTFS<Out, Uniform, FrameBuffer, then...>(cnt, tri, idx, uniform, frameBuffer, size, near, invnf);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applyTFS(unsigned int*, Triangle<Out>*,
                         TriangleRef*, Uniform*, FrameBuffer*, unsigned int,
                         float, float) {}

template <typename Out, typename Uniform, typename FrameBuffer,
          FSF<Out, Uniform, FrameBuffer>... fs>
CALLABLE void renderTrianglesGPU(unsigned int* cnt, Triangle<Out>* tri,
                                 TriangleRef* idx, Uniform* uniform, FrameBuffer* frameBuffer, const unsigned int size,
                                 const float near, const float invnf) {
    if (cnt[6]) {
        constexpr auto block = 1024U;
        run(cutTriangles<Out>, block, cnt[6], cnt, idx + size * 6, idx + size * 7);
        cudaDeviceSynchronize();
    }

    applyTFS<Out, Uniform, FrameBuffer, fs...>(cnt, tri, idx, uniform, frameBuffer, size, near, invnf);
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
    auto cnt = buffer.allocBuffer<unsigned int>(9);
    buffer.memset(cnt);
    auto info = buffer.allocBuffer<Triangle<Out>>(psiz);
    auto idx = buffer.allocBuffer<TriangleRef>(psiz * 10);
    vec2 fsize = size - uvec2{ 1, 1 };
    auto hfsize = static_cast<vec2>(size) * 0.5f;
    buffer.runKernelLinear(processTriangles<Index, Out, Uniform, cs>, index.size(), vert.get(),
                           index, uniform.get(),near,far,
        buffer.makeLazyConstructor<TriangleProcessingArgs<Out>>(cnt, info, idx,fsize,hfsize, psiz,
            static_cast<int>(mode)));

    if (history.enableSelfAdaptiveAllocation) {
        LaunchSize triNumData(cnt,8);
        triNumData.download(history.triNum, buffer);
    }

    //pass 2:render triangles
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderTrianglesGPU<Out, Uniform, FrameBuffer, fs...>, cnt, info, idx, uniform.get(),
                      frameBuffer.get(), psiz, near, invnf);
}
