#pragma once
#include <Base/CompileBegin.hpp>
#include <math_functions.h>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>
#include <ScanLineRenderer/Vertex.hpp>

template <typename Out>
struct LineInfo final {
    unsigned int id;
    VertexInfo<Out> a, b;
};

struct LineRef final {
    unsigned int id,size;
    vec4 rect;
    vec2 range;//begin,len
};

template <typename Out, typename Uniform, typename FrameBuffer>
using FSFL = void(*)(unsigned int id, ivec2 uv, float z, const Out& in,
                     const Uniform& uniform, FrameBuffer& frameBuffer);

CUDAINLINE auto calcTileSize(const vec2 a, const vec2 b) {
    return static_cast<unsigned int>(fmin(11.0f,ceil(log2f(distance(a,b)))));
}

template <typename Out>
GLOBAL void processLines(const unsigned int size,READONLY(VertexInfo<Out>) in,
                         LineInfo<Out>* info, LineRef* ref,unsigned int* cnt, const vec2 fsiz,const vec2 hsiz,
    const float near, const float far) {
    const auto id = getID();
    if (id >= size)return;
    auto a = in[id << 1], b = in[id << 1 | 1];
    if (a.pos.z > b.pos.z)std::swap(a, b);
    const auto pa = a.pos, pb = b.pos;
    if (pa.z >= far | pb.z <= near)return;
    if (pa.z < near)a = lerpZ(a, b, near);
    if (pb.z > far)b = lerpZ(a, b, far);
    a.pos = toRaster(a.pos,hsiz);
    b.pos = toRaster(b.pos, hsiz);
    const vec4 rect = { fmax(0.0f,fmin(a.pos.x,b.pos.x)),fmin(fsiz.x,fmax(a.pos.x,b.pos.x)),
        fmax(0.0f,fmin(a.pos.y,b.pos.y)),fmin(fsiz.y,fmax(a.pos.y,b.pos.y)) };
    if(rect.x<=rect.y && rect.z<=rect.w) {
        const auto p = atomicInc(cnt+12,maxv);
        LineInfo<Out> out;
        out.id = id;
        out.a.pos = a.pos;
        out.a.out =a.out*a.pos.z;
        out.b.pos = b.pos;
        out.b.out =b.out*b.pos.z;
        info[p] = out;
        LineRef res;
        res.id = p;
        res.size = calcTileSize(vec2(rect.x, rect.z), vec2(rect.y, rect.w));
        res.rect = rect;
        res.range = {0.0f,1.0f};
        ref[p] = res;
        atomicInc(cnt + res.size, maxv);
    }
}

std::pair<MemoryRef<unsigned int>, MemoryRef<LineRef>> sortLines(CommandBuffer& buffer,
    const MemoryRef<unsigned int>& cnt, const MemoryRef<LineRef>& ref);

//1...1024
template <typename Out, typename Uniform, typename FrameBuffer,
    FSFL<Out, Uniform, FrameBuffer> fs>
    GLOBAL void drawMicroL(READONLY(LineInfo<Out>) info, READONLY(LineRef) idx,
        READONLY(Uniform) uniform, FrameBuffer* frameBuffer,
        const float near, const float invnf) {
    const auto ref = idx[blockIdx.x];
    const auto line = info[ref.id];
    const auto w = ref.range.x + ref.range.y*threadIdx.x / blockDim.x;
    vec2 weight = { w,1.0f - w };
    const auto p = line.a.pos*weight.x + line.b.pos*weight.y;
    const auto z = 1.0f / p.z;
    weight *= z;
    const auto fout =line.a.out*weight.x+line.b.out*weight.y;
    fs(line.id, ivec2{ p.x,p.y }, (z-near)*invnf, fout, *uniform, *frameBuffer);
}

template <typename Out, typename Uniform, typename FrameBuffer,
    FSFL<Out, Uniform, FrameBuffer> first, FSFL<Out, Uniform, FrameBuffer>... then>
    CUDAINLINE void applyLFS(unsigned int* offset, LineInfo<Out>* tri, LineRef* idx,
        Uniform* uniform,FrameBuffer* frameBuffer,const float near,const float invnf) {
    for (auto i = 0; i < 11; ++i) {
        const auto size = offset[i + 1] - offset[i];
        if (size) {
            const auto bsiz = 1U << i;
            dim3 grid(size);
            dim3 block(bsiz);
            drawMicroL<Out, Uniform, FrameBuffer, first> << <grid, block >> >(tri, idx + offset[i],
                uniform, frameBuffer, near, invnf);
        }
    }

    cudaDeviceSynchronize();
    applyLFS<Out, Uniform, FrameBuffer, then...>(offset, tri, idx, uniform, frameBuffer, near, invnf);
}

template <typename Out, typename Uniform, typename FrameBuffer>
CUDAINLINE void applyLFS(unsigned int*, LineInfo<Out>*, LineRef*, Uniform*, FrameBuffer*,
    const float,const float) {}

template <typename Out, typename Uniform, typename FrameBuffer,
    FSFL<Out, Uniform, FrameBuffer>... fs>
    GLOBAL void renderLinesGPU(unsigned int* offset, LineInfo<Out>* tri, LineRef* idx,
        Uniform* uniform, FrameBuffer* frameBuffer, const float near, const float invnf) {
    applyLFS<Out, Uniform, FrameBuffer, fs...>(offset, tri, idx, uniform, frameBuffer, near, invnf);
}

template <typename Out, typename Uniform, typename FrameBuffer,
          FSFL<Out, Uniform, FrameBuffer>... fs>
void renderLines(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
                 const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer, const uvec2 size,
                 const float near, const float far) {
    auto cnt = buffer.allocBuffer<unsigned int>(13);
    buffer.memset(cnt);
    auto lsiz = vert.size() / 2;
    auto info = buffer.allocBuffer<LineInfo<Out>>(lsiz);
    auto ref = buffer.allocBuffer<LineRef>(lsiz);
    const vec2 fsiz = size - uvec2{ 1, 1 };
    const auto hsiz = static_cast<vec2>(size) * 0.5f;
    buffer.runKernelLinear(processLines<Out>,lsiz,vert,info,ref,cnt,fsiz,hsiz,near,far);
    auto sortedLines = sortLines(buffer,cnt,ref);
    cnt.earlyRelease();
    ref.earlyRelease();
    const auto invnf = 1.0f / (far - near);
    buffer.callKernel(renderLinesGPU<Out,Uniform,FrameBuffer,fs...>,sortedLines.first,info,
        sortedLines.second,uniform.get(),frameBuffer.get(),near,invnf);
}
