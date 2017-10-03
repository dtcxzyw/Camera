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
    cudaMemsetAsync(cnt.begin(), 0, sizeof(unsigned int), pipeline.getId());
    auto info = allocBuffer<Triangle<Out>>(index.size());
    pipeline.run(clipTriangles<Out>, index.size(),cnt.begin(),pos.begin(),out.begin()
        ,index.begin(), info.begin());
    pipeline.sync();
    auto num = *cnt.begin();
    if (num) {
        constexpr auto tileSize = 32U, clipSize = 2U, range = tileSize*clipSize;
        auto clipTileX = calcSize(size.x, range), clipTileY = calcSize(size.y, range);
        auto tcnt = allocBuffer<unsigned int>(clipTileX*clipTileY);
        auto tid = allocBuffer<unsigned int>(num*tcnt.size());
        std::vector<unsigned int> tsiz(tcnt.size());
        {
            cudaMemsetAsync(tcnt.begin(), 0, sizeof(unsigned int)*tcnt.size(), pipeline.getId());
            dim3 grid(num);
            dim3 block(clipTileX, clipTileY);
            pipeline.runDim(clipTile<Out>, grid,block, info.begin(), tcnt.begin(), tid.begin(), range);
            pipeline.sync();
            cudaMemcpy(tsiz.data(), tcnt.begin(), sizeof(unsigned int)*tcnt.size(), cudaMemcpyDefault);
        }

        for (unsigned int i = 0; i < clipTileX; ++i)
            for (unsigned int j = 0; j < clipTileY; ++j) {
                auto id = i*clipTileY + j;
                if (tsiz[id]) {
                    dim3 grid(tsiz[id], clipSize, clipSize);
                    dim3 block(tileSize, tileSize);
                    pipeline.runDim(drawTriangles<Out, Uniform, FrameBuffer, fs>, grid, block
                        ,info.begin(), tid.begin()+num*id, uniform.begin(), frameBuffer.begin()
                        , ivec2(i*range, j*range));
                }
            }
    }
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform,FrameBuffer> fs>
void renderFullScreen(Pipeline& pipeline,DataViewer<Uniform> uniform, 
    DataViewer<FrameBuffer> frameBuffer, uvec2 size) {
    pipeline.run(runFSFS<Uniform,FrameBuffer,fs>, size.x*size.y, uniform.begin(), frameBuffer.begin());
}

