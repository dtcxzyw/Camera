#pragma once
#include "ScanLine.hpp"
#include <thrust/sort.h>

template<typename Out>
struct Pack final {
    vec4 begin, end;
    Out bin, ein;
};

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void scanPointKernel(Pack<Out>* data
        , Uniform* uniform, FrameBuffer* frameBuffer, float begin, float len) {
    auto w = begin + getID() / len;
    fs(data->begin*(1.0f - w) + data->end*w, data->bin*(1.0f - w) + data->ein*w
        , 1.0f, *uniform, *frameBuffer);
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDA void drawScanLine(vec4 begin, vec4 end, const Out& bin, const Out& ein
        , Pack<Out>* p, Uniform& uniform, FrameBuffer& frameBuffer) {
    auto pixel = frameBuffer.size().x;
    float bx = clamp(begin.x, 0.0f, 1.0f), ex = clamp(end.x, 0.0f, 1.0f);
    p->begin = begin;
    p->end = end;
    p->bin = bin;
    p->ein = ein;
    auto line = end.x - begin.x;
    auto bw = (bx - begin.x) / line;
    auto ew = (ex - begin.x) / line;
    auto len = ew - bw;
    unsigned int size = fabs(ex - bx)*pixel;
    runGPU(scanPointKernel<Out, Uniform, FrameBuffer, fs>, size, p,
        &uniform, &frameBuffer, bw, len*size);
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void scanLineKernel(const vec4* pos, const Out* out, Uniform* uniform
        , FrameBuffer* frameBuffer, Pack<Out>* buffer, float begin, float len) {
    auto id = getID();
    auto w = begin + id / len;
    auto b = pos[0] * (1.0f - w) + pos[1] * w;
    auto e = pos[0] * (1.0f - w) + pos[2] * w;
    auto bin = out[0] * (1.0f - w) + out[1] * w;
    auto ein = out[0] * (1.0f - w) + out[2] * w;
    drawScanLine<Out, Uniform, FrameBuffer, fs>(b, e, bin, ein, buffer + id, *uniform, *frameBuffer);
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDA void scanTriangle(const vec4* pos, const Out* out, Uniform& uniform
        , FrameBuffer& frameBuffer, bool flag) {
    float top = pos[flag].y, bottom = pos[!flag].y;
    if (top > 1.0f || bottom < 0.0f)
        return;
    auto dtop = fmax(top, 0.0f), dbottom = fmin(bottom, 1.0f);
    auto pixel = frameBuffer.size().y;
    auto line = bottom - top;
    auto begin = (dtop - top) / line;
    auto end = (dbottom - top) / line;
    auto len = end - begin;
    unsigned int size = (dbottom - dtop)*pixel;
    auto p = static_cast<Pack<Out>*>(malloc(sizeof(Pack<Out>)*size));
    runGPU(scanLineKernel<Out, Uniform, FrameBuffer, fs>, size,
        pos, out, &uniform, &frameBuffer, p, begin, len*size);
    free(p);
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDA void drawTriangle(vec4* pos, Out* out, Uniform& uniform
        , FrameBuffer& frameBuffer) {
    vec4 tmpp = pos[2];
    Out tmpo = out[2];
    auto w = (pos[1].y - pos[0].y) / (pos[2].y - pos[0].y);
    pos[2] = pos[0] * (1.0f - w) + tmpp * w;
    out[2] = out[0] * (1.0f - w) + tmpo*w;
    scanTriangle<Out, Uniform, FrameBuffer, fs>(pos, out, uniform, frameBuffer, false);//0 1 mid
    pos[0] = tmpp;
    out[0] = tmpo;
    scanTriangle<Out, Uniform, FrameBuffer, fs>(pos, out, uniform, frameBuffer, true);//2 1 mid
}

template<typename Out>
struct Buffer final {
    vec4 pos[3];
    Out out[3];
};

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawTriangles(vec4* pos, Out* out, uvec3* index, Uniform* uniform,
        FrameBuffer* frameBuffer, Buffer<Out>* buffer) {
    auto id = getID();
    auto t = index[id];
    unsigned int idx[3] = { 0,1,2 };
    thrust::sort(idx, idx + 3, [&](auto a, auto b) {return pos[t[a]].y < pos[t[b]].y; });
    vec4* tsp = buffer[id].pos;
    Out* tso = buffer[id].out;
    for (auto i = 0; i < 3; ++i)
        tsp[i] = pos[t[idx[i]]], tso[i] = out[t[idx[i]]];
    drawTriangle<Out, Uniform, FrameBuffer, fs>(tsp, tso, *uniform, *frameBuffer);
}

