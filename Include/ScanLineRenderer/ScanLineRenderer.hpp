#pragma once
#include <Base/Pipeline.hpp>
#include "ScanLine.hpp"
#include "LineRasterizer.hpp"
#include "TriangleRasterizer.hpp"

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
auto calcVertex(Stream& stream, DataViewer<Vert> vert, const Uniform* uniform, uvec2 size) {
    auto vertex = allocBuffer<VertexInfo<Out>>(vert.size());
    stream.run(runVS<Vert, Out, Uniform, vs>, vert.size(),
        vert.begin(), uniform, vertex.begin(), static_cast<vec2>(size));
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
    void renderPoints(Stream& stream, DataViewer<VertexInfo<Out>> vert,
        const Uniform* uniform, FrameBuffer* frameBuffer, uvec2 size) {
    stream.run(drawPointHelper<Out, Uniform, FrameBuffer, ds>, vert.size(), vert.begin(), uniform, frameBuffer, size);
    stream.run(drawPointHelper<Out, Uniform, FrameBuffer, fs>, vert.size(), vert.begin(), uniform, frameBuffer, size);
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
    void renderTriangles(Stream& stream, DataViewer<VertexInfo<Out>> vert,
        Index index, const Uniform* uniform, FrameBuffer* frameBuffer, uvec2 size) {
    auto cnt = allocBuffer<unsigned int>(8);
    stream.memset(cnt);
    auto info = allocBuffer<Triangle<Out>>(index.size());
    stream.run(clipTriangles<Index, Out>, index.size(), cnt.begin(), vert.begin(), index,
        info.begin(), static_cast<vec2>(size));
    stream.sync();
    unsigned int triNum[7],offset[7];
    for (auto i = 0; i < 7; ++i)
        triNum[i] = cnt[i];
    auto all = cnt[7];
    if (all == 0)return;
    auto tri = allocBuffer<Triangle<Out>>(all);
    auto off = 0U;
    for (int i = 0; i < 7; ++i) {
        offset[i] = off;
        off += triNum[i];
    }
    auto head = stream.share(offset);
    stream.run(sortTriangles<Out>, all, info.begin(),tri.begin(),head.begin());

    for (auto i = 0; i < 6; ++i)
        if (triNum[i]) {
            auto bsiz = 1U << i;
            dim3 grid(triNum[i]);
            dim3 block(bsiz, bsiz);
            stream.runDim(drawMicroT<Out, Uniform, FrameBuffer, ds>, grid, block,tri.begin()+offset[i],
                uniform, frameBuffer);
        }

    if (triNum[6]) {
        auto num = triNum[6];
        auto clipTileX = calcSize(size.x, tileSize), clipTileY = calcSize(size.y, tileSize);
        auto tcnt = allocBuffer<unsigned int>(clipTileX*clipTileY);
        auto tid = allocBuffer<unsigned int>(num*tcnt.size());
        auto ptri =  tri.begin() +offset[6];
        dim3 block(clipTileX, clipTileY);
        {
            stream.memset(tcnt);
            dim3 grid(num);
            stream.runDim(clipTile<Out>, grid, block, ptri, tcnt.begin(), tid.begin());
        }
        {
            dim3 grid;
            stream.runDim(drawTile<Out, Uniform, FrameBuffer, ds>, grid, block, tcnt.begin(), ptri,
                tid.begin(), uniform, frameBuffer, num);
            stream.runDim(drawTile<Out, Uniform, FrameBuffer, fs>, grid, block, tcnt.begin(), ptri,
                tid.begin(), uniform, frameBuffer, num);
        }
    }

    for (auto i = 0; i < 6; ++i)
        if (triNum[i]) {
            auto bsiz = 1U << i;
            dim3 grid(triNum[i]);
            dim3 block(bsiz, bsiz);
            stream.runDim(drawMicroT<Out, Uniform, FrameBuffer, fs>, grid, block, tri.begin() + offset[i],
                uniform, frameBuffer);
        }
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform, FrameBuffer> fs>
void renderFullScreen(Stream& stream, const Uniform* uniform,
    FrameBuffer frameBuffer, uvec2 size) {
    stream.run(runFSFS<Uniform, FrameBuffer, fs>, size.x*size.y, uniform, frameBuffer, size.x);
}

