#pragma once
#include <Base/Pipeline.hpp>
#include <ScanLineRenderer/ScanLine.hpp>
#include <ScanLineRenderer/TriangleRasterizater.hpp>

template<typename Vert, typename Out, typename Uniform, typename FrameBuffer,
    VSF<Vert, Out, Uniform> vs, FSF<Out, Uniform, FrameBuffer> fs
    ,FSF<Out,Uniform,FrameBuffer> ds>
 void renderTriangles(Pipeline& pipeline,DataViewer<Vert> vert, DataViewer<uvec3> index,
        const Uniform* uniform, FrameBuffer* frameBuffer,uvec2 size) {
    auto vertex = allocBuffer<VertexInfo<Out>>(vert.size());
    pipeline.run(runVS<Vert, Out, Uniform,vs>, vert.size(),
        vert.begin(), uniform,vertex.begin(),static_cast<vec2>(size));
    auto cnt = allocBuffer<unsigned int>(7);
    cudaMemsetAsync(cnt.begin(), 0, sizeof(unsigned int)*cnt.size(), pipeline.getId());
    auto info = allocBuffer<Triangle<Out>>(7*index.size());
    pipeline.run(clipTriangles<Out>, index.size(),cnt.begin(),vertex.begin(),index.begin(),
        info.begin(),static_cast<vec2>(size));
    pipeline.sync();
    unsigned int microNum[6];
    for (auto i = 0; i < 6; ++i)
        microNum[i] = cnt[i];
    auto num = cnt[6];

    for (auto i = 0; i<6; ++i)
        if (microNum[i]) {
            dim3 grid(microNum[i]);
            dim3 block(1 << i, 1 << i);
            auto tri = info.begin() + i*index.size();
            pipeline.runDim(drawMicro<Out, Uniform, FrameBuffer, ds>, grid, block, tri,
                uniform, frameBuffer);
        }

    for (auto i = 0; i<6; ++i)
        if (microNum[i]) {
            dim3 grid(microNum[i]);
            dim3 block(1 << i, 1 << i);
            auto tri = info.begin() + i*index.size();
            pipeline.runDim(drawMicro<Out, Uniform, FrameBuffer, fs>, grid, block, tri,
                uniform, frameBuffer);
        }

    if (num) {
        auto clipTileX = calcSize(size.x, range), clipTileY = calcSize(size.y, range);
        auto tcnt = allocBuffer<unsigned int>(clipTileX*clipTileY);
        auto tid = allocBuffer<unsigned int>(num*tcnt.size());
        auto tri = info.begin() + 6 * index.size();
        {
            cudaMemsetAsync(tcnt.begin(), 0, sizeof(unsigned int)*tcnt.size(), pipeline.getId());
            dim3 grid(num);
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(clipTile<Out>, grid,block, tri, tcnt.begin(), tid.begin(), range);
        }
        {
            dim3 grid;
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(drawTile<Out, Uniform, FrameBuffer, ds>, grid, block, tcnt.begin(),tri,
                tid.begin(), uniform, frameBuffer, num);
        }
        {
            dim3 grid;
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(drawTile<Out, Uniform, FrameBuffer, fs>,grid,block,tcnt.begin(), tri,
                tid.begin(), uniform, frameBuffer,num);
        }
    }
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform,FrameBuffer> fs>
void renderFullScreen(Pipeline& pipeline,const Uniform* uniform, 
    FrameBuffer frameBuffer, uvec2 size) {
    pipeline.run(runFSFS<Uniform,FrameBuffer,fs>, size.x*size.y, uniform, frameBuffer,size.x);
}

