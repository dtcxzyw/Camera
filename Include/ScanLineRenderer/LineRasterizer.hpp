#pragma once
#include "ScanLine.hpp"
#include <math_functions.hpp>
#include <device_atomic_functions.h>

template<typename Out>
struct Line final {
    vec3 pa, pb;
    vec4 rect;
    Out oa, ob;
};

template<typename Out>
CALLABLE void sortLines(unsigned int size, unsigned int* cnt,
    const VertexInfo<Out>* ReadOnly vert, Line<Out>* info, vec2 fsize) {
    auto id = getID();
    if (id >= size)return;
    auto a = id << 1, b = id << 1 | 1;
    if (vert[a].flag | vert[b].flag == 0b111111) {
        Line<Out> res;
        res.pa = vert[a].pos, res.pb = vert[b].pos;
        res.rect = { fmax(0.0f,fmin(res.pa.x,res.pb.x)),fmin(fsize.x,fmax(res.pa.x,res.pb.x)),
            fmax(0.0f,fmin(res.pa.y,res.pb.y)),fmin(fsize.y,fmax(res.pa.y,res.pb.y)) };
        res.oa = vert[a].out, res.ob = vert[b].out;
        auto len = distance(res.pa, res.pb);
        auto x = static_cast<int>(ceil(log2f(fmin(len + 1.0f, 300.0f))));
        info[x*size + atomicInc(cnt + x, maxv)] = res;
    }
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDAInline void drawPoint(Line<Out> line, float w, Uniform uni, FrameBuffer& frameBuffer) {
    auto p = line.pa*w + line.pb*(1.0f - w);
    if (p.z >= 0.0f & p.z <= 1.0f) {
        auto fo = line.oa*w + line.ob*(1.0f - w);
        fs(p, p.z, fo, uni, frameBuffer);
    }
}

//1...256
template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawMicroL(const Line<Out>* ReadOnly info,
        const Uniform* ReadOnly uniform, FrameBuffer* frameBuffer, float len) {
    auto line = info[blockIdx.x];
    auto w = threadIdx.x / len;
    drawPoint<Out,Uniform,FrameBuffer,fs>(line, w, *uniform, *frameBuffer);
}

