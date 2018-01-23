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

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawPointHelperGPU(unsigned int size, ReadOnlyCache(VertexInfo<Out>) vert,
        ReadOnlyCache(Uniform) uniform, FrameBuffer* frameBuffer, vec2 fsize,
        float near,float invnf,vec2 hfsize) {
    auto id = getID();
    if (id >= size)return;
    auto p = vert[id];
    auto nz = (p.pos.z - near)*invnf;
    p.pos = toRaster(p.pos, hfsize.x,hfsize.y);
    if (0.0f <= p.pos.x & p.pos.x <= fsize.x & 0.0f <= p.pos.y & p.pos.y <= fsize.y
        & 0.0f <= nz & nz <= 1.0f) {
        fs(id,{ p.pos.x,p.pos.y },nz, p.out, *uniform, *frameBuffer);
    }
}

template< typename Out, typename Uniform, typename FrameBuffer>
    void drawPointHelper(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer, vec2 fsize, 
        float near, float invnf,vec2 hfsize){}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> first,FSF<Out,Uniform,FrameBuffer>... then>
    void drawPointHelper(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer, vec2 fsize,
        float near, float invnf,vec2 hfsize) {
    buffer.runKernelLinear(drawPointHelperGPU<Out, Uniform, FrameBuffer, first>, vert.size(), vert,
        uniform,frameBuffer, fsize, near, invnf, hfsize);
    drawPointHelper<Out,Uniform,FrameBuffer,then...>(buffer,vert,uniform,frameBuffer,fsize,
        near,invnf,hfsize);
}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer>... fs>
    void renderPoints(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, FrameBuffer* frameBuffer, uvec2 size,float near,float far) {
    vec2 fsize = size - uvec2{1, 1};
    auto hfsize = static_cast<vec2>(size) * 0.5f;
    auto invnf = 1.0f / (far - near);
    drawPointHelper<Out, Uniform, FrameBuffer, fs...>(buffer,vert,uniform,frameBuffer,fsize,
        near,invnf,hfsize);
}

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
    TCSF<Uniform> cs,FSF<Out, Uniform, FrameBuffer>... fs>
    void renderTriangles(CommandBuffer& buffer,const DataPtr<VertexInfo<Out>>& vert,
        Index index,const DataPtr<Uniform>& uniform,const DataPtr<FrameBuffer>& frameBuffer,
        uvec2 size,float near,float far,TriangleRenderingHistory& history,CullFace mode=CullFace::Back) {
    //pass 1:cull faces
    auto triNum = buffer.allocBuffer<unsigned int>();
    buffer.memset(triNum);
    auto tsiz = calcBufferSize(history.triNum,history.baseSize,index.size());
    auto vertBuffer = buffer.allocBuffer<TriangleVert<Out>>(tsiz);
    buffer.runKernelLinear(clipTriangles<Index, Out,Uniform,cs>, index.size(), triNum, vert.get(),
        index,uniform.get(), vertBuffer);
    LaunchSize triNumData(triNum);
    triNumData.download(history.triNum,buffer);

    //pass 2:clipping
    auto triNear = clipVertT<Out,compareZNear>(buffer, vertBuffer,LaunchSize(triNum),near,tsiz);
    triNum.earlyRelease(), vertBuffer.earlyRelease();

    auto triFar = clipVertT<Out, compareZFar>(buffer, triNear.second, triNear.first, far, tsiz);
    triNear.second.earlyRelease();

    //pass 3:process triangles
    vec2 fsize = size - uvec2{ 1,1 };
    auto psiz = calcBufferSize(history.processSize,history.baseSize,tsiz);
    auto cnt =buffer.allocBuffer<unsigned int>(9);
    buffer.memset(cnt);
    auto info =buffer.allocBuffer<Triangle<Out>>(psiz);
    auto idx = buffer.allocBuffer<TriangleRef>(psiz*10);
    auto hfsize = static_cast<vec2>(size)*0.5f;
    buffer.callKernel(processTrianglesGPU<Out>, triFar.first.get(), cnt, triFar.second, info,idx,
        fsize.x,fsize.y,hfsize.x,hfsize.y, psiz,static_cast<int>(mode));
    LaunchSize processSizeData(cnt, 8);
    processSizeData.download(history.processSize, buffer);
    //pass 4:render triangles
    auto invnf =1.0f/(far - near);
    buffer.callKernel(renderTrianglesGPU<Out, Uniform,FrameBuffer,fs...>,cnt,info,idx,uniform.get(),
        frameBuffer.get(), psiz,near,invnf);
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
