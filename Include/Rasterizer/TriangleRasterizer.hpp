#pragma once
#include <Rasterizer/Vertex.hpp>
#include <Rasterizer/Shared.hpp>
#include <Base/CompileBegin.hpp>
#include <math_functions.h>
#include <Base/CompileEnd.hpp>
#include <Rasterizer/Tile.hpp>

///@see <a href="https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.compatibility.pdf">OpenGL 4.6 API Specification, Section 10.1 Primitive Types</a>

class TriangleStrips final {
public:
    static auto size(const unsigned int vertSize) {
        return vertSize - 2;
    }

    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        return {off, off + 1, off + 2};
    }
};

class TriangleFans final {
public:
    static auto size(const unsigned int vertSize) {
        return vertSize - 2;
    }

    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        return {0, off + 1, off + 2};
    }
};

class SeparateTriangles final {
public:
    static auto size(const unsigned int faceSize) {
        return faceSize;
    }

    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        const auto base = off * 3;
        return {base, base + 1, base + 2};
    }
};

class SeparateTrianglesWithIndex final {
private:
    READONLY(uvec3) mPtr;
public:
    explicit SeparateTrianglesWithIndex(READONLY(uvec3) idx): mPtr(idx) {}

    static auto size(const unsigned int faceSize) {
        return faceSize;
    }

    CUDAINLINE auto operator[](const unsigned int off) const {
        return mPtr[off];
    }
};

template <typename Out>
struct STRUCT_ALIGN TriangleVert final {
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
struct STRUCT_ALIGN Triangle final {
    vec3 invz;
    mat3 w;
    unsigned int id, type;
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
        const vec4 iScissor, const vec2 iHsiz, const int iMode, const unsigned int iMaxSize)
        : cnt(iCnt), info(iInfo), out(iOut), scissor(iScissor), hsiz(iHsiz), mode(iMode), maxSize(iMaxSize) {}
};

template <typename Out>
CUDAINLINE void calcTriangleInfo(const TriangleVert<Out>& tri, const TriangleProcessingArgs<Out>& args) {
    const auto a = toRaster(tri.vert[0].pos, args.hsiz),
        b = toRaster(tri.vert[1].pos, args.hsiz),
        c = toRaster(tri.vert[2].pos, args.hsiz);
    const uvec4 rect = {
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
        res.type = ((c.y == b.y & c.x > b.x) | (c.y > b.y)) |
            ((a.y == c.y & a.x > c.x) | (a.y > c.y)) << 1 |
            ((b.y == a.y & b.x > a.x) | (b.y > a.y)) << 2;
        res.out[0] = tri.vert[0].out * res.invz.x;
        res.out[1] = tri.vert[1].out * res.invz.y;
        res.out[2] = tri.vert[2].out * res.invz.z;
        TileRef ref;
        ref.size = calcTileSize(rect);
        ref.rect = rect;
        atomicInc(args.cnt + ref.size, maxv);
        ref.id = atomicInc(args.cnt + 6, maxv);
        if (ref.id >= args.maxSize)return;
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
CUDAINLINE void sortIndex(const TriangleVert<Out>& tri, uvec3& idx) {
    if (Func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
    if (Func(tri.vert[idx[2]].pos.z, tri.vert[idx[1]].pos.z))cudaSwap(idx[2], idx[1]);
    if (Func(tri.vert[idx[1]].pos.z, tri.vert[idx[0]].pos.z))cudaSwap(idx[1], idx[0]);
}

template <typename Out, CompareZ Func, typename Callable>
CUDAINLINE void clipVertT1(const TriangleVert<Out>& tri, const float z, const Callable& emit) {
    uvec3 idx{0, 1, 2};
    sortIndex<Out, Func>(tri, idx);
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
CUDAINLINE void clipVertT2(const TriangleVert<Out>& tri, const float z, const Callable& emit) {
    uvec3 idx{0, 1, 2};
    sortIndex<Out, Func>(tri, idx);
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
CUDAINLINE void clipTriangle(const TriangleVert<Out>& tri, const float z, const Callable& emit) {
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
using TriangleClipShader = bool(*)(unsigned int id, vec3& pa, vec3& pb, vec3& pc, const Uniform& uniform);

template <typename Index, typename Out, typename Uniform, TriangleClipShader<Uniform> Func>
GLOBAL void processTrianglesKernel(const unsigned int size,READONLY(VertexInfo<Out>) vert,
    Index index,READONLY(Uniform) uniform, const float near, const float far,
    TriangleProcessingArgs<Out> args) {
    const auto id = getId();
    if (id >= size)return;
    const auto idx = index[id];
    TriangleVert<Out> tri;
    tri.id = id, tri.vert[0] = vert[idx[0]], tri.vert[1] = vert[idx[1]], tri.vert[2] = vert[idx[2]];
    if (Func(tri.id, tri.vert[0].pos, tri.vert[1].pos, tri.vert[2].pos, *uniform)) {
        const auto emitF = [&args](const TriangleVert<Out>& t) {
            calcTriangleInfo<Out>(t, args);
        };
        const auto emitN = [far,&emitF](const TriangleVert<Out>& t) {
            clipTriangle<Out, compareZFar, decltype(emitF)>(t, far, emitF);
        };
        clipTriangle<Out, compareZNear, decltype(emitN)>(tri, near, emitN);
    }
}

CUDAINLINE bool calcWeight(const mat4 w0, const int type, const vec2 p, const vec3 invz,
    const float near, const float invnf, vec3& w, float& nz) {
    #pragma unroll
    for (auto i = 0; i < 3; ++i)
        w[i] = w0[i].x * p.x + w0[i].y * p.y + w0[i].z;
    //w /= w.x + w.y + w.z;
    const bool flag = (w.x == 0.0f ? type : w.x > 0.0f) &
        (w.y == 0.0f ? type >> 1 : w.y > 0.0f) &
        (w.z == 0.0f ? type >> 2 : w.z > 0.0f) & 1;
    const auto z = 1.0f / (invz.x * w.x + invz.y * w.y + invz.z * w.z);
    w.x *= z, w.y *= z, w.z *= z;
    nz = (z - near) * invnf;
    return flag;
}

template <typename Out, typename Uniform, typename FrameBuffer>
using FragmentShader = void(*)(unsigned int id, ivec2 uv, float z, const Out& in,
    const Out& ddx, const Out& ddy, const Uniform& uniform, FrameBuffer& frameBuffer);

//2,4,8,16,32
template <typename Out, typename Uniform, typename FrameBuffer,
    FragmentShader<Out, Uniform, FrameBuffer> FragShader>
GLOBAL void drawMicroT(READONLY(Triangle<Out>) info,READONLY(TileRef) idx,
    READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
    const float near, const float invnf, const float sx, const float sy) {
    const auto& ref = idx[blockIdx.x];
    const auto offX = threadIdx.x >> 1U, offY = threadIdx.x & 1U;
    const ivec2 uv{ref.rect.x + (threadIdx.y << 1) + offX, ref.rect.z + (threadIdx.z << 1) + offY};
    const vec2 p{uv.x + sx, uv.y + sy};
    const auto& tri = info[ref.id];
    vec3 w;
    float z;
    const auto flag = calcWeight(tri.w, tri.type, p, tri.invz, near, invnf, w, z);
    const auto ddx = (shuffleVec3(w, 0b10) - w) * (offX ? -1.0f : 1.0f);
    const auto ddy = (shuffleVec3(w, 0b01) - w) * (offY ? -1.0f : 1.0f);
    if (uv.x <= ref.rect.y & uv.y <= ref.rect.w & flag) {
        const auto fout = tri.out[0] * w.x + tri.out[1] * w.y + tri.out[2] * w.z;
        const auto ddxo = tri.out[0] * ddx.x + tri.out[1] * ddx.y + tri.out[2] * ddx.z;
        const auto ddyo = tri.out[0] * ddy.x + tri.out[1] * ddy.y + tri.out[2] * ddy.z;
        FragShader(tri.id, uv, z, fout, ddxo, ddyo, *uniform, *frameBuffer);
    }
}

template <typename Out, typename Uniform, typename FrameBuffer,
    FragmentShader<Out, Uniform, FrameBuffer> Func, FragmentShader<Out, Uniform, FrameBuffer>... Then>
CUDAINLINE void applyTFS(unsigned int* offset, Triangle<Out>* tri, TileRef* idx, Uniform* uniform,
    FrameBuffer* frameBuffer, const float near, const float invnf, const vec2 samplePoint) {
    #pragma unroll
    for (auto i = 0; i < 5; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            const dim3 grid(size);
            const auto bsiz = 1 << i;
            const dim3 block(4, bsiz, bsiz);
            drawMicroT<Out, Uniform, FrameBuffer, Func> << <grid, block >> >(tri, idx + offset[i],
                uniform, frameBuffer, near, invnf, samplePoint.x, samplePoint.y);
        }
    }

    cudaDeviceSynchronize();
    applyTFS<Out, Uniform, FrameBuffer, Then...>(offset, tri, idx, uniform, frameBuffer, near, invnf,
        samplePoint);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applyTFS(unsigned int*, Triangle<Out>*, TileRef*, Uniform*, FrameBuffer*,
    float, float, vec2) {}

template <typename Out, typename Uniform, typename FrameBuffer,
    FragmentShader<Out, Uniform, FrameBuffer>... FragShader>
GLOBAL void renderTrianglesKernel(unsigned int* offset, Triangle<Out>* tri, TileRef* idx,
    Uniform* uniform, FrameBuffer* frameBuffer, const float near, const float invnf,
    const vec2 samplePoint) {
    applyTFS<Out, Uniform, FrameBuffer, FragShader...>(offset, tri, idx, uniform, frameBuffer, near, invnf,
        samplePoint);
}

template<typename Uniform>
using TriangleTileClipShader = bool(*)(const uvec4& rect, const Uniform& uniform);

template<typename Uniform>
CUDAINLINE bool emptyTriangleTileClipShader(const uvec4&, const Uniform&) {
    return true;
}

CUDAINLINE bool intersect(const mat3& w0,const vec4& rect) {
    const auto test=[&rect](const vec3 w){
        return fmax(rect.x*w.x, rect.y*w.x) + fmax(rect.z*w.y, rect.w*w.y) + w.z >= 0.0;
    };
    return test(w0[0]) & test(w0[1]) & test(w0[2]);
}

template<typename Out,typename Uniform,TriangleTileClipShader<Uniform> Func>
CUDAINLINE bool triangleTileClipShader(const TileRef& ref,const Uniform& uni,
    READONLY(Triangle<Out>) data) {
    return intersect(data[ref.id].w,ref.rect) && Func(ref.rect,uni);
}

template<typename Out>
struct CachedTriangleProcessingResult final {
    MemorySpan<unsigned int> cnt;
    MemorySpan<Triangle<Out>> info;
    MemorySpan<TileRef> idx;
    unsigned int maxSize;
};

template<typename Out>
struct TriangleProcessingResult final {
    Span<unsigned int> cnt;
    Span<Triangle<Out>> info;
    Span<TileRef> idx;
    unsigned int maxSize;
    bool useCache;
    TriangleProcessingResult(const Span<unsigned int>& cnt, const Span<Triangle<Out>>& info,
        const Span<TileRef>& idx, const unsigned int maxSize,const bool useCache) 
        : cnt(cnt), info(info), idx(idx), maxSize(maxSize), useCache(useCache) {}
};

template<typename Out, typename Judge = EmptyJudge>
struct TriangleRenderingContext final : Uncopyable {
    unsigned int baseSize, renderCount;
    PinnedBuffer<unsigned int> triNum;
    bool enableSelfAdaptiveAllocation;

    using JudgeType = And<EqualComparer<vec4>, Judge>;
    SharedCacheHolder<std::shared_ptr<CachedTriangleProcessingResult<Out>>, JudgeType>
        processCache;
    bool enableCache;

    TriangleRenderingContext() : baseSize(0), renderCount(0), triNum(1),
        enableSelfAdaptiveAllocation(false), enableCache(false) {}

    unsigned int calcBufferSize(const unsigned int maxv) const {
        return baseSize + std::min(*triNum + (*triNum >> 3), maxv);
    }

    void reset(const unsigned int size, const unsigned int base,
        const bool SAA, const bool cache) {
        *triNum = size;
        baseSize = base;
        enableSelfAdaptiveAllocation = SAA;
        enableCache = cache;
    }
};

template <typename IndexDesc, typename Out, typename Uniform,
    TriangleClipShader<Uniform> ClipShader, typename Judge>
auto processTriangles(CommandBuffer& buffer,
    const Span<VertexInfo<Out>>& vert, const IndexDesc& index,
    const Span<Uniform>& uniform, const uvec2 size, const float near, const float far,
    const vec4& scissor, const CullFace mode, TriangleRenderingContext<Out, Judge>& context,
    const Judge& judge, const size_t psiz) {
    using namespace std::chrono_literals;
    using Result= TriangleProcessingResult<Out>;
    using CacheType = CachedTriangleProcessingResult<Out>;
    using JudgeType = typename TriangleRenderingContext<Out, Judge>::JudgeType;
    using CacheRefType = CacheRef<std::shared_ptr<CacheType>, JudgeType>;
    const auto mergedJudge = makeEqualComparer(scissor) && judge;
    CacheRefType ref;
    if (context.enableCache)ref = CacheRefType{ context.processCache, mergedJudge };
    else context.processCache.reset();
    if (ref && ref.vaild()) {
        auto val = ref.getValue();
        return Result(buffer.useAllocated(val->cnt), buffer.useAllocated(val->info),
            buffer.useAllocated(val->idx), val->maxSize, true);
    }
    std::shared_ptr<CacheType> newCache;
    if (context.enableCache) {
        newCache = std::shared_ptr<CacheType>(new CacheType, [&context, mergedJudge]
        (CacheType* ptr) {
            context.processCache = std::make_shared<CachedValueHolder<std::shared_ptr<CacheType>,
                JudgeType>>(std::shared_ptr<CacheType>(ptr), mergedJudge);
        });
    }
    MemoryReleaseFunction func;
    if (context.enableCache) {
        func = [=](UniqueMemory memory, size_t memSize) {
            newCache->cnt = MemorySpan<unsigned int>{ std::move(memory), memSize };
        };
    }
    //5+ext+cnt=7
    auto cnt = buffer.allocBuffer<unsigned int>(7, std::move(func));
    buffer.memset(cnt);
    if (context.enableCache) {
        func = [=](UniqueMemory memory, size_t memSize) {
            newCache->info = MemorySpan<Triangle<Out>>{ std::move(memory), memSize };
        };
    }
    auto info = buffer.allocBuffer<Triangle<Out>>(psiz, std::move(func));
    if (context.enableCache) {
        func = [=](UniqueMemory memory, size_t memSize) {
            newCache->idx = MemorySpan<TileRef>{ std::move(memory), memSize };
        };
    }
    auto idx = buffer.allocBuffer<TileRef>(psiz, std::move(func));
    const auto hfsize = static_cast<vec2>(size) * 0.5f;
    const unsigned int maxSize = std::min(info.maxSize(), idx.maxSize());
    if (context.enableCache)newCache->maxSize = maxSize;
    buffer.launchKernelLinear(processTrianglesKernel<IndexDesc::IndexType, Out, Uniform, ClipShader>,
        index.size(), vert, index.get(), uniform, near, far,
        buffer.makeLazyConstructor<TriangleProcessingArgs<Out>>(cnt, info, idx, scissor, hfsize,
            static_cast<int>(mode), maxSize));
    return Result(cnt, info, idx, maxSize, false);
}

template <typename IndexDesc, typename Out, typename Uniform, typename FrameBuffer,
    TriangleClipShader<Uniform> ClipShader,TriangleTileClipShader<Uniform> TileClipShader, 
    typename Judge, FragmentShader<Out, Uniform, FrameBuffer>... FragShader>
void renderTriangles(CommandBuffer& buffer, const Span<VertexInfo<Out>>& vert,
    const IndexDesc& index, const Span<Uniform>& uniform, const Span<FrameBuffer>& frameBuffer,
    const uvec2 size, const float near, const float far,vec4 scissor,
    TriangleRenderingContext<Out, Judge>& context, const Judge& judge = {},
    const CullFace mode = CullFace::Back, const vec2 samplePoint = {0.5f, 0.5f}) {
    ++context.renderCount;
    scissor = {
        fmax(samplePoint.x, scissor.x),
        fmin(size.x - 1 + samplePoint.x, scissor.y),
        fmax(samplePoint.y, scissor.z),
        fmin(size.y - 1 + samplePoint.y, scissor.w)
    };
    //pass 1:process triangles
    const auto psiz = context.calcBufferSize(index.size());
    auto res = processTriangles<IndexDesc, Out, Uniform, ClipShader, Judge>(buffer, vert, index,
        uniform, size, near, far, scissor, mode, context, judge, psiz);
    //pass 2:sort triangles
    auto sortedTri = sortTiles<Uniform, Triangle<Out>, 
        triangleTileClipShader<Out, Uniform, TileClipShader>>(buffer, res.cnt, res.idx, 
            res.maxSize * 2, res.maxSize, uniform, res.info);
    if ((context.renderCount & 31) == 0 && !res.useCache && 
        context.enableSelfAdaptiveAllocation) {
        const LaunchSize triNumData(res.cnt.subSpan(6));
        triNumData.download(*context.triNum, buffer);
    }

    //pass 3:render triangles
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderTrianglesKernel<Out, Uniform, FrameBuffer, FragShader...>, 
        sortedTri.cnt, res.info, sortedTri.array, uniform, frameBuffer, near, invnf, 
        samplePoint);
}
