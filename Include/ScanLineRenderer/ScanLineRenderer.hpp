#pragma once
#include <Base/Pipeline.hpp>
#include <ScanLineRenderer/ScanLine.hpp>
#include <ScanLineRenderer/TriangleRasterizater.hpp>

template<typename Vert, typename Out, typename Uniform, typename FrameBuffer,
    VSF<Vert, Out, Uniform> vs, FSF<Out, Uniform, FrameBuffer> fs
    ,FSF<Out,Uniform,FrameBuffer> ds>
 void renderTriangles(Pipeline& pipeline,DataViewer<Vert> vert, DataViewer<uvec3> index,
        DataViewer<Uniform> uniform, DataViewer<FrameBuffer> frameBuffer,uvec2 size) {
    auto vertex = allocBuffer<VertexInfo<Out>>(vert.size());
    pipeline.run(runVS<Vert, Out, Uniform,FrameBuffer,vs,fs,ds>, vert.size(),
        vert.begin(), uniform.begin(),vertex.begin(),frameBuffer.begin());
    auto cnt = allocBuffer<unsigned int>(1);
    cudaMemsetAsync(cnt.begin(), 0, sizeof(unsigned int), pipeline.getId());
    auto info = allocBuffer<Triangle<Out>>(index.size());
    pipeline.run(clipTriangles<Out>, index.size(),cnt.begin(),vertex.begin(),index.begin(),
        info.begin(),static_cast<vec2>(size));
    pipeline.sync();
    auto num = *cnt.begin();
    if (num) {
        auto clipTileX = calcSize(size.x, range), clipTileY = calcSize(size.y, range);
        auto tcnt = allocBuffer<unsigned int>(clipTileX*clipTileY);
        auto tid = allocBuffer<unsigned int>(num*tcnt.size());
        {
            cudaMemsetAsync(tcnt.begin(), 0, sizeof(unsigned int)*tcnt.size(), pipeline.getId());
            dim3 grid(num);
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(clipTile<Out>, grid,block, info.begin(), tcnt.begin(), tid.begin(), range);
        }
        {
            dim3 grid;
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(drawTile<Out, Uniform, FrameBuffer, ds>, grid, block, tcnt.begin(), info.begin(),
                tid.begin(), uniform.begin(), frameBuffer.begin(), num);
        }
        {
            dim3 grid;
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(drawTile<Out, Uniform, FrameBuffer, fs>,grid,block,tcnt.begin(), info.begin(),
                tid.begin(), uniform.begin(), frameBuffer.begin(),num);
        }
    }
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform,FrameBuffer> fs>
void renderFullScreen(Pipeline& pipeline,DataViewer<Uniform> uniform, 
    FrameBuffer frameBuffer, uvec2 size) {
    pipeline.run(runFSFS<Uniform,FrameBuffer,fs>, size.x*size.y, uniform.begin(), frameBuffer,size.x);
}

