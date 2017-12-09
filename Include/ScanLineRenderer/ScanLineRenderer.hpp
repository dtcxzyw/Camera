#pragma once
#include <Base/Pipeline.hpp>
#include <Base/DispatchSystem.hpp>
#include "ScanLine.hpp"
#include "LineRasterizer.hpp"
#include "TriangleRasterizer.hpp"

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
auto calcVertex(CommandBuffer& buffer, const MemoryRef<Vert>& vert
    , const MemoryRef<Uniform>& uniform, uvec2 size) {
    auto vertex = buffer.allocBuffer<VertexInfo<Out>>(vert.size());
    buffer.runKernelLinear(runVS<Vert, Out, Uniform, vs>, vert.size(),
        vert, uniform, vertex, static_cast<vec2>(size));
    return vertex;
}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawPointHelper(unsigned int size, const VertexInfo<Out>* ReadOnlyCache vert,
        const Uniform* ReadOnlyCache uniform, FrameBuffer* frameBuffer, uvec2 fsize) {
    auto id = getID();
    if (id >= size)return;
    auto p = vert[id];
    if (p.flag == 0b111111)
        fs({ p.pos.x,p.pos.y }, p.pos.z, p.out, *uniform, *frameBuffer);
}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderPoints(CommandBuffer& buffer, const MemoryRef<VertexInfo<Out>>& vert,
        const MemoryRef<Uniform>& uniform, FrameBuffer* frameBuffer, uvec2 size) {
    buffer.runKernelLinear(drawPointHelper<Out, Uniform, FrameBuffer, ds>, vert.size(), vert, uniform, frameBuffer, size);
    buffer.runKernelLinear(drawPointHelper<Out, Uniform, FrameBuffer, fs>, vert.size(), vert, uniform, frameBuffer, size);
}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderLines(Stream& stream, DataViewer<VertexInfo<Out>> vert,
        const Uniform* uniform, FrameBuffer* frameBuffer, uvec2 size) {
    auto cnt = allocBuffer<unsigned int>(11);
    stream.memset(cnt);
    auto lsiz = vert.size() / 2;
    auto info = allocBuffer<Line<Out>>(11 * lsiz);
    stream.run(sortLines<Out>, lsiz, cnt.begin(), vert.begin(), info.begin(), static_cast<vec2>(size));
    stream.sync();
    unsigned int lineNum[11];
    for (auto i = 0; i < 11; ++i)
        lineNum[i] = cnt[i];

    if (lineNum[10]) {
        auto num = lineNum[10];
        auto base = info.begin() + 9 * lsiz;
        stream.run(cutLines<Out>, num, &cnt[9], base + lsiz, base, size, 1 << 9);
        stream.sync();
        lineNum[9] = cnt[9];
    }

    for (auto i = 0; i < 10; ++i)
        if (lineNum[i]) {
            dim3 grid(lineNum[i]);
            dim3 block(1 << i);
            auto base = info.begin() + i*lsiz;
            stream.runDim(drawMicroL<Out, Uniform, FrameBuffer, ds>, grid, block, base,
                uniform, frameBuffer, 1 << i);
        }

    for (auto i = 0; i < 10; ++i)
        if (lineNum[i]) {
            dim3 grid(lineNum[i]);
            dim3 block(1 << i);
            auto base = info.begin() + i*lsiz;
            stream.runDim(drawMicroL<Out, Uniform, FrameBuffer, fs>, grid, block, base,
                uniform, frameBuffer, 1 << i);
        }
}

template<typename Index, typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderTriangles(CommandBuffer& buffer, const MemoryRef<VertexInfo<Out>>& vert,
        Index index, const MemoryRef<Uniform>& uniform, FrameBuffer* frameBuffer, uvec2 size) {
    auto cnt =buffer.allocBuffer<unsigned int>(8);
    buffer.memset(cnt);
    auto info =buffer.allocBuffer<Triangle<Out>>(index.size()*2);
    auto idx = buffer.allocBuffer<unsigned int>(index.size()*7);
    buffer.runKernelLinear(clipTriangles<Index, Out>, index.size(), cnt, vert, index,
        info,idx, static_cast<vec2>(size));
    buffer.callKernel(renderTrianglesGPU<Out, Uniform,FrameBuffer,ds,fs>,cnt,info,idx
        ,uniform,frameBuffer,index.size());
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform, FrameBuffer> fs>
void renderFullScreen(CommandBuffer& buffer, const Uniform* uniform,
    FrameBuffer frameBuffer, uvec2 size) {
    buffer.runKernelDim(runFSFS<Uniform, FrameBuffer, fs>, size.x*size.y, uniform, frameBuffer, size.x);
}
