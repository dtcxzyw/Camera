#pragma once
#include <Base/Pipeline.hpp>
#include <ScanLineRenderer/ScanLine.hpp>
#include <ScanLineRenderer/TriangleRasterizater.hpp>

template<typename Vert, typename Out, typename Uniform, typename FrameBuffer,
    VSF<Vert, Out, Uniform> vs, FSF<Out, Uniform, FrameBuffer> fs>
 void renderTriangles(Pipeline& pipeline,DataViewer<Vert> vert, DataViewer<uvec3> index,
        DataViewer<Uniform> uniform, DataViewer<FrameBuffer> frameBuffer,uvec2 size) {
    auto pos = allocBuffer<vec4>(vert.size());
    auto out = allocBuffer<Out>(vert.size());
    pipeline.run(runVS<Vert, Out, Uniform,FrameBuffer, vs>, vert.size(), vert.begin(), uniform.begin(),
        out.begin(), pos.begin(),frameBuffer.begin());
    auto cnt = allocBuffer<unsigned int>(1);
    pipeline.sync();
    *cnt.begin() = 0;
    auto info = allocBuffer<Triangle<Out>>(index.size());
    pipeline.run(clipTriangles<Out>, index.size(),cnt.begin(),pos.begin(),out.begin(),index.begin(), info.begin());
    pipeline.sync();
    constexpr auto tileSize = 32U,clipSize=2U,range=tileSize*clipSize;
    auto num = *cnt.begin();
    if (num) {
        auto clipTileX = calcSize(size.x, range), clipTileY = calcSize(size.y, range);
        auto now = allocBuffer<Triangle<Out>>(index.size());
        for (unsigned int i = 0; i < clipTileX; ++i)
            for (unsigned int j = 0; j < clipTileY; ++j) {
                vec4 rect = { i*range,(i + 1)*range,j*range,(j + 1)*range };
                pipeline.sync();
                *cnt.begin() = 0;
                pipeline.run(clipTile<Out>, num,info.begin(),cnt.begin(), now.begin(),rect);
                pipeline.sync();
                auto tcnt = *cnt.begin();
                if (tcnt) {
                    dim3 grid(tcnt, clipSize, clipSize);
                    dim3 block(tileSize, tileSize);
                    pipeline.runDim(drawTriangles<Out, Uniform, FrameBuffer, fs>, grid, block
                        , now.begin(), uniform.begin(), frameBuffer.begin(), ivec2(i*range, j*range));
                }
            }
    }
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform,FrameBuffer> fs>
void renderFullScreen(Pipeline& pipeline,DataViewer<Uniform> uniform, 
    DataViewer<FrameBuffer> frameBuffer, uvec2 size) {
    pipeline.run(runFSFS<Uniform,FrameBuffer,fs>, size.x*size.y, uniform.begin(), frameBuffer.begin());
}

