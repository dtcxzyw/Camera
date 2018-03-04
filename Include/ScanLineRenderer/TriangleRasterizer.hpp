#pragma once
#include <ScanLineRenderer/Vertex.hpp>
#include <ScanLineRenderer/Shared.hpp>
#include <Base/CompileBegin.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>
#include <cuda_device_runtime_api.h>
#include <sm_30_intrinsics.h>
#include <Base/CompileEnd.hpp>

///@see <a href="https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.compatibility.pdf">OpenGL 4.6 API Specification, Section 10.1 Primitive Types</a>

class TriangleStrips final {
private:
    unsigned int mSize;
public:
    explicit TriangleStrips(const unsigned int vertSize) : mSize(vertSize - 2) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        return {off, off + 1, off + 2};
    }
};

class TriangleFans final {
private:
    unsigned int mSize;
public:
    explicit TriangleFans(const unsigned int vertSize) : mSize(vertSize - 2) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        return {0, off + 1, off + 2};
    }
};

class SeparateTriangles final {
private:
    unsigned int mSize;
public:
    explicit SeparateTriangles(const unsigned int faceSize) : mSize(faceSize) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        const auto base = off * 3;
        return {base, base + 1, base + 2};
    }
};

class SeparateTrianglesWithIndex final {
private:
    READONLY(uvec3) mPtr;
    unsigned int mSize;
public:
    SeparateTrianglesWithIndex(READONLY(uvec3) idx, const unsigned int faceSize)
        : mPtr(idx), mSize(faceSize) {}

    explicit SeparateTrianglesWithIndex(const DataViewer<uvec3>& ibo)
        : SeparateTrianglesWithIndex(ibo.begin(), ibo.size()) {}

    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE auto operator[](const unsigned int off) const {
        return mPtr[off];
    }
};

template <typename Out>
struct TriangleVert final {
    unsigned int id;
    VertexInfo<Out> vert[3];
};

CUDAINLINE float edgeFunction(const vec3 a, const vec3 b, const vec3 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
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

enum class CullFace {
    Front = 0,
    Back = 1,
    None = 2
};

template <typename Out>
struct TriangleProcessingArgs final {
    unsigned int* cnt;
    Triangle<Out>* info;
    TileRef* out;
    vec4 scissor;
    vec2 hsiz;
    int mode;
    unsigned int maxSize;

    TriangleProcessingArgs(unsigned int* iCnt, Triangle<Out>* iInfo, TileRef* iOut,
                           const vec4 iScissor, const vec2 iHsiz, const int iMode,const unsigned int iMaxSize)
        : cnt(iCnt), info(iInfo), out(iOut), scissor(iScissor), hsiz(iHsiz), mode(iMode),maxSize(iMaxSize) {}
};

template <typename Out>
CUDAINLINE void calcTriangleInfo(const TriangleVert<Out>& tri, const TriangleProcessingArgs<Out>& args) {
    const auto a = toRaster(tri.vert[0].pos, args.hsiz),
               b = toRaster(tri.vert[1].pos, args.hsiz),
               c = toRaster(tri.vert[2].pos, args.hsiz);
    const vec4 rect = {
        fmax(args.scissor.x, min3(a.x, b.x, c.x) - tileOffset),
        fmin(args.scissor.y, max3(a.x, b.x, c.x) + tileOffset),
        fmax(args.scissor.z, min3(a.y, b.y, c.y) - tileOffset),
        fmin(args.scissor.w, max3(a.y, b.y, c.y) + tileOffset)
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
        TileRef ref;
        ref.size = calcTileSize(rect);
        atomicInc(args.cnt + ref.size, maxv);
        ref.id = atomicInc(args.cnt + 6, maxv);
        if(ref.id>=args.maxSize)return;
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
CUDAINLINE int calcTriangleType(const TriangleVert<Out>& tri, const float z) {
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
CUDAINLINE void clipVertT1(const TriangleVert<Out>& tri, const float z,const Callable& emit) {
    const auto idx = sortIndex<Out, Func>(tri, uvec3{0, 1, 2});
    const auto a = tri.vert[idx[0]], b = tri.vert[idx[1]], c = tri.vert[idx[2]];
    const auto d = lerpZ(b, a, z), e = lerpZ(c, a, z);
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
CUDAINLINE void clipVertT2(const TriangleVert<Out>& tri, const float z,const Callable& emit) {
    const auto idx = sortIndex<Out, Func>(tri, uvec3{0, 1, 2});
    const auto a = tri.vert[idx[0]], b = tri.vert[idx[1]], c = tri.vert[idx[2]];
    const auto d = lerpZ(a, c, z), e = lerpZ(b, c, z);
    TriangleVert<Out> out;
    out.id = tri.id;
    if ((idx[2] + 1) % 3 == idx[0])
        out.vert[0] = d, out.vert[1] = e, out.vert[2] = c;
    else
        out.vert[0] = e, out.vert[1] = d, out.vert[2] = c;

    emit(out);
}

template <typename Out, CompareZ Func, typename Callable>
CUDAINLINE void clipTriangle(const TriangleVert<Out>& tri, const float z,const Callable& emit) {
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
using TCSF = bool(*)(unsigned int idx, vec3& pa, vec3& pb, vec3& pc, const Uniform& uniform);

template <typename Index, typename Out, typename Uniform, TCSF<Uniform> cs>
GLOBAL void processTriangles(const unsigned int size,READONLY(VertexInfo<Out>) vert,
                             Index index,READONLY(Uniform) uniform, const float near, const float far,
                             TriangleProcessingArgs<Out> args) {
    const auto id = getID();
    if (id >= size)return;
    const auto idx = index[id];
    TriangleVert<Out> tri;
    tri.id = id, tri.vert[0] = vert[idx[0]], tri.vert[1] = vert[idx[1]], tri.vert[2] = vert[idx[2]];
    if (cs(id, tri.vert[0].pos, tri.vert[1].pos, tri.vert[2].pos, *uniform)) {
        const auto emitF = [&args](const TriangleVert<Out>& t) {
            calcTriangleInfo<Out>(t, args);
        };
        const auto emitN = [far,&emitF](const TriangleVert<Out>& t) {
            clipTriangle<Out, compareZFar, decltype(emitF)>(t, far, emitF);
        };
        clipTriangle<Out, compareZNear, decltype(emitN)>(tri, near, emitN);
    }
}

CUDAINLINE bool calcWeight(const mat4 w0, const vec2 p, const vec3 invz,
                           const float near, const float invnf, vec3& w, float& nz) {
    w.x = w0[0].x * p.x + w0[0].y * p.y + w0[0].z;
    w.y = w0[1].x * p.x + w0[1].y * p.y + w0[1].z;
    w.z = w0[2].x * p.x + w0[2].y * p.y + w0[2].z;
    const bool flag = w.x >= 0.0f & w.y >= 0.0f & w.z >= 0.0f;
    const auto z = 1.0f / (invz.x * w.x + invz.y * w.y + invz.z * w.z);
    w.x *= z, w.y *= z, w.z *= z;
    nz = (z - near) * invnf;
    return flag;
}

template <typename Out, typename Uniform, typename FrameBuffer>
using FSFT = void(*)(unsigned int id, ivec2 uv, float z, const Out& in,
                     const Out& ddx, const Out& ddy, const Uniform& uniform, FrameBuffer& frameBuffer);

//2,4,8,16,32
template <typename Out, typename Uniform, typename FrameBuffer,
    FSFT<Out, Uniform, FrameBuffer> fs>
GLOBAL void drawMicroT(READONLY(Triangle<Out>) info,READONLY(TileRef) idx,
                       READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
                       const float near, const float invnf) {
    const auto ref = idx[blockIdx.x];
    const auto offX = threadIdx.x >> 1U, offY = threadIdx.x & 1U;
    const ivec2 uv{ref.rect.x + (threadIdx.y << 1) + offX, ref.rect.z + (threadIdx.z << 1) + offY};
    const vec2 p{uv.x + 0.5f, uv.y + 0.5f};
    const auto tri = info[ref.id];
    vec3 w;
    float z;
    const auto flag = calcWeight(tri.w, p, tri.invz, near, invnf, w, z);
    const auto ddx = (shuffleVec3(w, 0b10) - w) * (offX ? -1.0f : 1.0f);
    const auto ddy = (shuffleVec3(w, 0b01) - w) * (offY ? -1.0f : 1.0f);
    if (p.x <= ref.rect.y & p.y <= ref.rect.w & flag) {
        const auto fout = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        const auto ddxo = tri.out[0] * ddx.x + tri.out[1] * ddx.y + tri.out[2] * ddx.z;
        const auto ddyo = tri.out[0] * ddy.x + tri.out[1] * ddy.y + tri.out[2] * ddy.z;
        fs(tri.id, uv, z, fout, ddxo, ddyo, *uniform, *frameBuffer);
    }
}

template <typename Out, typename Uniform, typename FrameBuffer,
    FSFT<Out, Uniform, FrameBuffer> first, FSFT<Out, Uniform, FrameBuffer>... then>
CUDAINLINE void applyTFS(unsigned int* offset, Triangle<Out>* tri, TileRef* idx, Uniform* uniform,
                         FrameBuffer* frameBuffer, const float near, const float invnf) {
    #pragma unroll
    for (auto i = 0; i < 5; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            dim3 grid(size);
            const auto bsiz = 1 << i;
            dim3 block(4, bsiz, bsiz);
            drawMicroT<Out, Uniform, FrameBuffer, first> << <grid, block >> >(tri, idx + offset[i],
                                                                              uniform, frameBuffer, near, invnf);
        }
    }

    cudaDeviceSynchronize();
    applyTFS<Out, Uniform, FrameBuffer, then...>(offset, tri, idx, uniform, frameBuffer, near, invnf);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applyTFS(unsigned int*, Triangle<Out>*, TileRef*, Uniform*, FrameBuffer*,
                         float, float) {}

template <typename Out, typename Uniform, typename FrameBuffer,
    FSFT<Out, Uniform, FrameBuffer>... fs>
GLOBAL void renderTrianglesGPU(unsigned int* offset, Triangle<Out>* tri, TileRef* idx,
                               Uniform* uniform, FrameBuffer* frameBuffer, const float near, const float invnf) {
    applyTFS<Out, Uniform, FrameBuffer, fs...>(offset, tri, idx, uniform, frameBuffer, near, invnf);
}

struct TriangleRenderingHistory final : Uncopyable {
    unsigned int baseSize;
    PinnedBuffer<unsigned int> triNum;
    bool enableSelfAdaptiveAllocation;

    TriangleRenderingHistory() : baseSize(0), triNum(1), enableSelfAdaptiveAllocation(false) {}

    unsigned int calcBufferSize(const unsigned int maxv) const {
        return baseSize + std::min(*triNum + (*triNum >> 3), maxv);
    }

    void reset(const unsigned int size, const unsigned int base = 2048U, const bool SAA = false) {
        *triNum = size;
        baseSize = base;
        enableSelfAdaptiveAllocation = SAA;
    }
};

template <typename IndexDesc, typename Out, typename Uniform, typename FrameBuffer,
    TCSF<Uniform> cs, FSFT<Out, Uniform, FrameBuffer>... fs>
void renderTriangles(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
                     const IndexDesc& index, const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer,
                     const uvec2 size, const float near, const float far, TriangleRenderingHistory& history,
                     vec4 scissor,const CullFace mode = CullFace::Back) {
    //pass 1:process triangles
    const auto psiz = history.calcBufferSize(index.size());
    //5+ext+cnt=7
    auto cnt = buffer.allocBuffer<unsigned int>(7);
    buffer.memset(cnt);
    auto info = buffer.allocBuffer<Triangle<Out>>(psiz);
    auto idx = buffer.allocBuffer<TileRef>(psiz);
    scissor = { fmax(0.5f,scissor.x),fmin(size.x - 0.5f,scissor.y),
        fmax(0.5f,scissor.z),fmin(size.y - 0.5f,scissor.w) };
    const auto hfsize = static_cast<vec2>(size) * 0.5f;
    const unsigned int maxSize = std::min(info.maxSize(), idx.maxSize());
    buffer.runKernelLinear(processTriangles<IndexDesc::IndexType, Out, Uniform, cs>, index.size(),
        vert.get(), index.get(), uniform.get(), near, far,
        buffer.makeLazyConstructor<TriangleProcessingArgs<Out>>(cnt, info, idx, scissor, hfsize,
            static_cast<int>(mode),maxSize));

    //pass 2:sort triangles
    auto sortedTri = sortTiles(buffer, cnt, idx, psiz * 2, maxSize);
    if (history.enableSelfAdaptiveAllocation) {
        LaunchSize triNumData(cnt, 6);
        triNumData.download(*history.triNum, buffer);
    }
    cnt.earlyRelease();
    idx.earlyRelease();

    //pass 3:render triangles
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderTrianglesGPU<Out, Uniform, FrameBuffer, fs...>, sortedTri.first, info,
                      sortedTri.second, uniform.get(), frameBuffer.get(), near, invnf);
}
