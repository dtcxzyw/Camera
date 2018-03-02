#pragma once
#include <Base/CompileBegin.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>
#include <ScanLineRenderer/Vertex.hpp>
#include <ScanLineRenderer/Shared.hpp>

///@see <a href="https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.compatibility.pdf">OpenGL 4.6 API Specification, Section 10.1 Primitive Types</a>

class LineStrips final {
private:
    unsigned int mSize;
public:
    explicit LineStrips(const unsigned int vertSize) : mSize(vertSize - 1) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec2 operator[](const unsigned int off) const {
        return {off, off + 1};
    }
};

class LineLoops final {
private:
    unsigned int mSize;
public:
    explicit LineLoops(const unsigned int vertSize) : mSize(vertSize) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec2 operator[](const unsigned int off) const {
        const auto next = off + 1;
        return {off, next == mSize ? 0 : next};
    }
};

class SeparateLines final {
private:
    unsigned int mSize;
public:
    explicit SeparateLines(const unsigned int lineSize) : mSize(lineSize) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec2 operator[](const unsigned int off) const {
        const auto base = off << 1;
        return {base, base + 1};
    }
};

class SeparateTrianglesWireframe final {
private:
    unsigned int mSize;
public:
    explicit SeparateTrianglesWireframe(const unsigned int faceSize) : mSize(faceSize * 3) {}
    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec2 operator[](const unsigned int off) const {
        const auto tri = off / 3, id = off % 3, base = tri * 3;
        return {base + id, base + (id + 1) % 3};
    }
};

class SeparateTrianglesWithIndexWireframe final {
private:
    READONLY(uvec3) mPtr;
    unsigned int mSize;
public:
    SeparateTrianglesWithIndexWireframe(READONLY(uvec3) idx, const unsigned int faceSize)
        : mPtr(idx), mSize(faceSize * 3) {}

    explicit SeparateTrianglesWithIndexWireframe(const DataViewer<uvec3>& ibo)
        : SeparateTrianglesWithIndexWireframe(ibo.begin(), ibo.size()) {}

    BOTH auto size() const {
        return mSize;
    }

    CUDAINLINE uvec2 operator[](const unsigned int off) const {
        const auto tri = off / 3, id = off % 3;
        const auto idx = mPtr[tri];
        return {idx[id], idx[(id + 1) % 3]};
    }
};

template <typename Out>
struct LineInfo final {
    unsigned int id;
    VertexInfo<Out> a, b;
};

struct LineRef final {
    unsigned int id, size;
    float len;
    vec2 range; //begin,len
};

template <typename Out, typename Uniform, typename FrameBuffer>
using FSFL = void(*)(unsigned int id, ivec2 uv, float z, const Out& in,
                     const Uniform& uniform, FrameBuffer& frameBuffer);

CUDAINLINE auto calcTileSize(const float len) {
    return static_cast<unsigned int>(fmin(11.5f, ceil(log2f(len))));
}

CUDAINLINE vec2 calcRange(const float a, const float b, const float l,const float r) {
    if (b == a)return (l <= a & a <= r) ? vec2{ 0.0f, 1.0f } : vec2{ 0.0f, 0.0f };
    const auto invx = 1.0f / (b - a), lax = (l - a) * invx, rax = (r - a) * invx;
    return {fmin(lax, rax), fmax(lax, rax)};
}

CUDAINLINE vec2 calcLineRange(const vec2 a, const vec2 b, const vec4 scissor) {
    const auto rangeX = calcRange(a.x, b.x, scissor.x,scissor.y);
    const auto rangeY = calcRange(a.y, b.y, scissor.z,scissor.w);
    const auto begin = max3(0.0f, rangeX.x, rangeY.x);
    const auto end = min3(1.0f, rangeX.y, rangeY.y);
    return {begin, end - begin};
}

template <typename Index, typename Out, typename Uniform, PosConverter<Uniform> toPos>
GLOBAL void processLines(const unsigned int size,READONLY(VertexInfo<Out>) in, Index index,
                         LineInfo<Out>* info, LineRef* ref, unsigned int* cnt, const vec4 scissor, const vec2 hsiz,
                         const float near, const float far,READONLY(Uniform) uniform) {
    const auto id = getID();
    if (id >= size)return;
    auto idx = index[id];
    auto a = in[idx[0]], b = in[idx[1]];
    a.pos = toPos(a.pos, *uniform);
    b.pos = toPos(b.pos, *uniform);
    if (a.pos.z > b.pos.z)cudaSwap(a, b);
    if (a.pos.z >= far | b.pos.z <= near)return;
    if (a.pos.z < near)a = lerpZ(a, b, near);
    if (b.pos.z > far)b = lerpZ(a, b, far);
    a.pos = toRaster(a.pos, hsiz);
    b.pos = toRaster(b.pos, hsiz);
    const vec2 range = calcLineRange(a.pos, b.pos, scissor);
    if (range.y > 0.0f) {
        const auto p = atomicInc(cnt + 12, maxv);
        LineInfo<Out> out;
        out.id = id;
        out.a.pos = a.pos;
        out.a.out = a.out * a.pos.z;
        out.b.pos = b.pos;
        out.b.out = b.out * b.pos.z;
        info[p] = out;
        LineRef res;
        res.id = p;
        res.len = distance(vec2(a.pos), vec2(b.pos)) * range.y;
        res.size = calcTileSize(res.len);
        res.range = range;
        ref[p] = res;
        atomicInc(cnt + res.size, maxv);
    }
}

std::pair<MemoryRef<unsigned int>, MemoryRef<LineRef>> sortLines(CommandBuffer& buffer,
                                                                 const MemoryRef<unsigned int>& cnt,
                                                                 const MemoryRef<LineRef>& ref);

//1...1024
template <typename Out, typename Uniform, typename FrameBuffer,
    FSFL<Out, Uniform, FrameBuffer> fs>
GLOBAL void drawMicroL(READONLY(LineInfo<Out>) info, READONLY(LineRef) idx,
                       READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
                       const float near, const float invnf,
    const float bx, const float ex, const float by, const float ey) {
    const auto ref = idx[blockIdx.x];
    const auto line = info[ref.id];
    const auto w = ref.range.x + ref.range.y * threadIdx.x / blockDim.x;
    vec2 weight = {1.0f - w, w};
    const auto p = line.a.pos * weight.x + line.b.pos * weight.y;
    const auto z = 1.0f / p.z;
    weight *= z;
    const auto fout = line.a.out * weight.x + line.b.out * weight.y;
    if (bx <= p.x & p.x <= ex & by <= p.y & p.y <= ey)
        fs(line.id, ivec2{p.x, p.y}, (z - near) * invnf, fout, *uniform, *frameBuffer);
}

template <typename Out, typename Uniform, typename FrameBuffer,
    FSFL<Out, Uniform, FrameBuffer> first, FSFL<Out, Uniform, FrameBuffer>... then>
CUDAINLINE void applyLFS(unsigned int* offset, LineInfo<Out>* tri, LineRef* idx, Uniform* uniform,
                         FrameBuffer* frameBuffer, const float near, const float invnf, const vec4 scissor) {
    #pragma unroll
    for (auto i = 0; i < 11; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            const auto bsiz = 1U << i;
            dim3 grid(size);
            dim3 block(bsiz);
            drawMicroL<Out, Uniform, FrameBuffer, first> << <grid, block >> >(tri, idx + offset[i],
                uniform, frameBuffer, near, invnf, scissor.x, scissor.y, scissor.z, scissor.w);
        }
    }

    cudaDeviceSynchronize();
    applyLFS<Out, Uniform, FrameBuffer, then...>(offset, tri, idx, uniform, frameBuffer, near, invnf, scissor);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applyLFS(unsigned int*, LineInfo<Out>*, LineRef*, Uniform*, FrameBuffer*,
                         const float, const float, const vec4) {}

template <typename Out, typename Uniform, typename FrameBuffer,
    FSFL<Out, Uniform, FrameBuffer>... fs>
GLOBAL void renderLinesGPU(unsigned int* offset, LineInfo<Out>* tri, LineRef* idx,
                           Uniform* uniform, FrameBuffer* frameBuffer, const float near, const float invnf,
                           const vec4 scissor) {
    applyLFS<Out, Uniform, FrameBuffer, fs...>(offset, tri, idx, uniform, frameBuffer, near, invnf, scissor);
}

template <typename IndexDesc, typename Out, typename Uniform, typename FrameBuffer,
    PosConverter<Uniform> toPos, FSFL<Out, Uniform, FrameBuffer>... fs>
void renderLines(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,const IndexDesc& index,
                 const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer,
                 const uvec2 size, const float near, const float far,vec4 scissor) {
    auto cnt = buffer.allocBuffer<unsigned int>(13);
    buffer.memset(cnt);
    const auto lsiz = index.size();
    auto info = buffer.allocBuffer<LineInfo<Out>>(lsiz);
    auto ref = buffer.allocBuffer<LineRef>(lsiz);
    scissor = { fmax(0.5f,scissor.x),fmin(size.x - 0.5f,scissor.y),
        fmax(0.5f,scissor.z),fmin(size.y - 0.5f,scissor.w) };
    const auto hsiz = static_cast<vec2>(size) * 0.5f;
    buffer.runKernelLinear(processLines<IndexDesc::IndexType, Out, Uniform, toPos>, lsiz, vert.get(),
        index.get(), info, ref, cnt, scissor, hsiz, near, far, uniform.get());
    auto sortedLines = sortLines(buffer, cnt, ref);
    cnt.earlyRelease();
    ref.earlyRelease();
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderLinesGPU<Out, Uniform, FrameBuffer, fs...>, sortedLines.first, info,
                      sortedLines.second, uniform.get(), frameBuffer.get(), near, invnf, scissor);
}
