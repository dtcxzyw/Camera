#pragma once
#include "ScanLine.hpp"
#include <math_functions.h>
#include <device_atomic_functions.h>

template<typename Out>
struct Line final {
    vec3 pa, pb;
    vec4 rect;
    Out oa, ob;
};

template<typename Out>
CALLABLE void sortLines(unsigned int size, unsigned int* cnt,
    const VertexInfo<Out>* ReadOnlyCache vert, Line<Out>* info, vec2 fsize) {
    auto id = getID();
    if (id >= size)return;
    auto a = id << 1, b = id << 1 | 1;
    if (vert[a].flag | vert[b].flag == 0b111111) {
        Line<Out> res;
        res.pa = vert[a].pos, res.pb = vert[b].pos;
        res.rect = { fmax(0.0f,fmin(res.pa.x,res.pb.x)),fmin(fsize.x,fmax(res.pa.x,res.pb.x)),
            fmax(0.0f,fmin(res.pa.y,res.pb.y)),fmin(fsize.y,fmax(res.pa.y,res.pb.y)) };
        res.oa = vert[a].out, res.ob = vert[b].out;
        auto len = distance(vec2(res.pa), vec2(res.pb));
        auto x = static_cast<int>(ceil(log2f(fmin(len + 1.0f, 700.0f))));
        info[x*size + atomicInc(cnt + x, maxv)] = res;
    }
}

template<typename Out>
CUDAInline void calcX(Line<Out>& line, float x) {
    auto k = (line.pa.y - line.pb.y) / (line.pa.x - line.pb.x);
    auto b =line.pa.y-line.pa.x*k;
    auto y = k*x + b;
    auto len = distance2(vec2(line.pa), vec2(line.pb));
    auto dis = distance2(vec2(line.pa),vec2(x,y));
    auto w = sqrtf(dis/len);
    line.pa = { x,y, line.pa.z*(1.0f - w) + line.pb.z*w };
    line.oa = line.oa*(1.0f - w) + line.ob*w;
}

template<typename Out>
CUDAInline void calcY(Line<Out>& line, float y) {
    auto k = (line.pa.x - line.pb.x) / (line.pa.y - line.pb.y);
    auto b = line.pa.x - line.pa.y*k;
    auto x = k*y + b;
    auto len = distance2(vec2(line.pa), vec2(line.pb));
    auto dis = distance2(vec2(line.pa), vec2(x, y));
    auto w = sqrtf(dis / len);
    line.pa = { x,y, line.pa.z*(1.0f - w) + line.pb.z*w };
    line.oa = line.oa*(1.0f - w) + line.ob*w;
}

template<typename Out>
CALLABLE void cutLines(unsigned int size, unsigned int* cnt,
    const Line<Out>* ReadOnlyCache data, Line<Out>* out, vec2 fsize, float len) {
    auto id = getID();
    if (id >= size)return;
    auto line=data[id];
    calcX(line,fmax(0.0f,fmin(line.pa.x,fsize.x)));
    calcY(line, fmax(0.0f, fmin(line.pa.y, fsize.y)));
    swap(line.pa, line.pb);
    swap(line.oa, line.ob);
    calcX(line, fmax(0.0f, fmin(line.pa.x, fsize.x)));
    calcY(line, fmax(0.0f, fmin(line.pa.y, fsize.y)));
    auto clen= distance(vec2(line.pa), vec2(line.pb));
    float delta = len/clen;
    for (float w = 0.0f; w < 1.0f;w+=delta) {
        Line<Out> res;
        res.pa = res.pa*(1.0f - w) + res.pb*w;
        res.oa = res.oa*(1.0f - w) + res.ob*w;
        float wb = w + delta;
        res.pb = res.pa*(1.0f - wb) + res.pb*wb;
        res.ob = res.oa*(1.0f - wb) + res.ob*wb;
        out[atomicInc(cnt, maxv)] = res;
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

//1...512
template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawMicroL(const Line<Out>* ReadOnlyCache info,
        const Uniform* ReadOnlyCache uniform, FrameBuffer* frameBuffer, float len) {
    auto line = info[blockIdx.x];
    auto w = threadIdx.x / len;
    drawPoint<Out,Uniform,FrameBuffer,fs>(line, w, *uniform, *frameBuffer);
}

