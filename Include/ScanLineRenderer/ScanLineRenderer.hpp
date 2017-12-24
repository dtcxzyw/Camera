#pragma once
#include <Base/DispatchSystem.hpp>
#include "ScanLine.hpp"
#include "LineRasterizer.hpp"
#include "TriangleRasterizer.hpp"

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
auto calcVertex(CommandBuffer& buffer, const DataPtr<Vert>& vert
    , const DataPtr<Uniform>& uniform, uvec2 size) {
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
    void renderPoints(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, FrameBuffer* frameBuffer, uvec2 size) {
    buffer.runKernelLinear(drawPointHelper<Out, Uniform, FrameBuffer, ds>, vert.size(), vert, uniform, frameBuffer, size);
    buffer.runKernelLinear(drawPointHelper<Out, Uniform, FrameBuffer, fs>, vert.size(), vert, uniform, frameBuffer, size);
}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderLines(CommandBuffer& buffer,const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform,const DataPtr<FrameBuffer>& frameBuffer, uvec2 size) {
    auto cnt = buffer.allocBuffer<unsigned int>(12);
    buffer.memset(cnt);
    auto lsiz = vert.size() / 2;
    auto info = buffer.allocBuffer<Line<Out>>(lsiz);
    auto idx = buffer.allocBuffer<unsigned int>(lsiz * 11);
    buffer.runKernelLinear(sortLines<Out>, lsiz, cnt, vert,info, static_cast<vec2>(size));
    buffer.callKernel(renderLineGPU<Out, Uniform, FrameBuffer, ds, fs>,info,uniform,frameBuffer
        ,cnt,idx,lsiz, static_cast<vec2>(size));
}

template<typename Index, typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderTriangles(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        Index index, const DataPtr<Uniform>& uniform
        ,const DataPtr<FrameBuffer>& frameBuffer, uvec2 size) {
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
void renderFullScreen(CommandBuffer& buffer, const DataPtr<Uniform>& uniform,
    const Value<FrameBuffer>& frameBuffer, uvec2 size) {
    buffer.runKernelLinear(runFSFS<Uniform, FrameBuffer, fs>, size.x*size.y, uniform, frameBuffer, size.x);
}
