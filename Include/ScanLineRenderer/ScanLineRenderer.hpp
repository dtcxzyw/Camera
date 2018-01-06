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
    if (0.0f<=p.pos.x & p.pos.x<=fsize.x & 0.0f<=p.pos.y & p.pos.y<=fsize.y 
        & 0.0f<=p.pos.z & p.pos.z<=1.0f)
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
        Index index, const DataPtr<Uniform>& uniform,const DataPtr<FrameBuffer>& frameBuffer,
        uvec2 size) {
    auto cnt =buffer.allocBuffer<unsigned int>(9);
    buffer.memset(cnt);
    auto info =buffer.allocBuffer<Triangle<Out>>(std::max(65536U,index.size()*2U));
    auto idx = buffer.allocBuffer<unsigned int>(index.size()*10);
    buffer.runKernelLinear(clipTriangles<Index, Out>, index.size(), cnt, vert, index,
        info, idx, static_cast<vec2>(size) - vec2{1.0f, 1.0f});
    buffer.callKernel(renderTrianglesGPU<Out, Uniform,FrameBuffer,ds,fs>,cnt,info,idx
        ,uniform,frameBuffer,index.size());
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform, FrameBuffer> fs>
void renderFullScreen(CommandBuffer& buffer, const DataPtr<Uniform>& uniform,
    const Value<FrameBuffer>& frameBuffer, uvec2 size) {
    constexpr auto tileSize = 32U;
    dim3 grid(calcSize(size.x, tileSize), calcSize(size.y,tileSize));
    dim3 block(tileSize, tileSize);
    buffer.runKernelDim(runFSFS<Uniform, FrameBuffer, fs>,grid,block, uniform, frameBuffer, size);
}
