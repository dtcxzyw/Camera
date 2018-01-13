#pragma once
#include <Base/DispatchSystem.hpp>
#include "ScanLine.hpp"
//#include "LineRasterizer.hpp"
#include "TriangleRasterizer.hpp"

template<typename Vert, typename Out, typename Uniform,VSF<Vert, Out, Uniform> vs>
auto calcVertex(CommandBuffer& buffer,const DataPtr<Vert>& vert,const DataPtr<Uniform>& uniform) {
    auto vertex = buffer.allocBuffer<VertexInfo<Out>>(vert.size());
    buffer.runKernelLinear(runVS<Vert, Out, Uniform,vs>, vert.size(),
        vert.get(), uniform.get(), vertex);
    return vertex;
}

/*
template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawPointHelper(unsigned int size, ReadOnlyCache(VertexInfo<Out>) vert,
        ReadOnlyCache(Uniform) uniform, FrameBuffer* frameBuffer, uvec2 fsize,
        float near,float far,float invnf) {
    auto id = getID();
    if (id >= size)return;
    auto p = vert[id];
    if (0.0f<=p.pos.x & p.pos.x<=fsize.x & 0.0f<=p.pos.y & p.pos.y<=fsize.y 
        & near<=p.pos.z & p.pos.z<=far)
        fs({ p.pos.x,p.pos.y }, p.pos.z, p.out, *uniform, *frameBuffer);
}
*/

/*
template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderPoints(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, FrameBuffer* frameBuffer, uvec2 size,float near,float far) {
    vec2 fsize = size - uvec2{1, 1};
    auto invnf = 1.0f / (far - near);
    buffer.runKernelLinear(drawPointHelper<Out, Uniform, FrameBuffer, ds>, vert.size(), vert, uniform,
        frameBuffer, fsize,near,far,invnf);
    buffer.runKernelLinear(drawPointHelper<Out, Uniform, FrameBuffer, fs>, vert.size(), vert, uniform, 
        frameBuffer, fsize,near,far,invnf);
}
*/

/*
template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderLines(CommandBuffer& buffer,const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform,const DataPtr<FrameBuffer>& frameBuffer, uvec2 size,
        float near,float far) {
    vec2 fsize = size - uvec2{1, 1};
    auto cnt = buffer.allocBuffer<unsigned int>(12);
    buffer.memset(cnt);
    auto lsiz = vert.size() / 2;
    auto info = buffer.allocBuffer<Line<Out>>(lsiz);
    auto idx = buffer.allocBuffer<unsigned int>(lsiz * 11);
    auto invnf = 1.0f / (far - near);
    buffer.runKernelLinear(sortLines<Out>, lsiz, cnt, vert,info, fsize,near,far);
    buffer.callKernel(renderLineGPU<Out, Uniform, FrameBuffer, ds, fs>,info,uniform,frameBuffer
        ,cnt,idx,lsiz, fsize,near,invnf);
}
*/

template<typename Index, typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    void renderTriangles(CommandBuffer& buffer,const DataPtr<VertexInfo<Out>>& vert,
        Index index,const DataPtr<Uniform>& uniform,const DataPtr<FrameBuffer>& frameBuffer,
        uvec2 size,float near,float far,vec2 mul, CullFace mode = CullFace::Back) {
    vec2 fsize = size - uvec2{ 1,1 };
    //pass 1:cull faces
    auto triNum = buffer.allocBuffer<unsigned int>(1);
    buffer.memset(triNum);
    auto clipedVert = buffer.allocBuffer<TriangleVert<Out>>(index.size());
    buffer.runKernelLinear(clipTriangles<Index, Out>, index.size(), triNum, vert.get(), index, clipedVert,
        static_cast<int>(mode));
    //pass 2:clipping
    auto tsiz = std::max(65536U, index.size() * 2U);
    auto triNear = clipVertT<Out,compareZNear>(buffer,clipedVert,LaunchSize(triNum),near,tsiz);
    auto triFar = clipVertT<Out, compareZFar>(buffer, triNear.second, triNear.first, far, tsiz);
    //pass 3:process triangles
    auto cnt =buffer.allocBuffer<unsigned int>(9);
    buffer.memset(cnt);
    auto info =buffer.allocBuffer<Triangle<Out>>(tsiz);
    auto idx = buffer.allocBuffer<unsigned int>(tsiz*10);
    auto hfsize = static_cast<vec2>(size)*0.5f;
    buffer.callKernel(processTrianglesGPU<Out>, triFar.first.get(), cnt, triFar.second, info,idx,
        fsize.x,fsize.y,hfsize.x,hfsize.y,mul.x,mul.y);
    //pass 4:render triangles
    auto invnf =1.0f/(far - near);
    buffer.callKernel(renderTrianglesGPU<Out, Uniform,FrameBuffer,ds,fs>,cnt,info,idx,uniform.get(),
        frameBuffer.get(),index.size(),near,invnf);
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform, FrameBuffer> fs>
void renderFullScreen(CommandBuffer& buffer,const DataPtr<Uniform>& uniform,
    const Value<FrameBuffer>& frameBuffer, uvec2 size) {
    constexpr auto tileSize = 32U;
    dim3 grid(calcSize(size.x, tileSize), calcSize(size.y,tileSize));
    dim3 block(tileSize, tileSize);
    buffer.runKernelDim(runFSFS<Uniform, FrameBuffer, fs>,grid,block, uniform.get(),
        frameBuffer.get(), size);
}
