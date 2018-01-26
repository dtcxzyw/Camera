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
    READONLY(VertexInfo<Out>) vert, Line<Out>* info,unsigned int* lineID,vec2 fsize,
    float near,float far) {
    auto id = getID();
    if (id >= size)return;
    auto a = id << 1, b = id << 1 | 1;
    auto pa = vert[a].pos, pb = vert[b].pos;
    auto rect = { fmax(0.0f,fmin(pa.x,pb.x)),fmin(fsize.x,fmax(pa.x,pb.x)),
        fmax(0.0f,fmin(pa.y,pb.y)),fmin(fsize.y,fmax(pa.y,pb.y)) };
    float minz = fmax(near, fmin(pa.z, pb.z)), maxz = fmin(far, fmax(pa.z, pb.z));
    if (rect.x<rect.y & rect.z<rect.w & minz<=maxz) {
        pa.z = 1.0f / pa.z, pb.z = 1.0f / pb.z;
        Line<Out> res;
        res.rect = rect;
        res.pa = pa, res.pb = pb;
        res.oa = vert[a].out, res.ob = vert[b].out;
        auto len = distance2(vec2(res.pa), vec2(res.pb));
        auto x = static_cast<int>(ceil(0.5f*log2f(fmin(len + 1.0f, 5e5f))));
        auto lid = atomicInc(cnt + 11, maxv);
        info[lid] = res;
        lineID[x*size + atomicInc(cnt + x, maxv)] = lid;
    }
}

template<typename Out>
CUDAINLINE void calcX(Line<Out>& line, float x) {
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
CUDAINLINE void calcY(Line<Out>& line, float y) {
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
CALLABLE void cutLines(unsigned int size, unsigned int* cnt,unsigned int* wp,
    READONLY(Line<Out>) data,unsigned int* iidx,unsigned int* oidx, vec2 fsize, float len) {
    auto id = getID();
    if (id >= size)return;
    auto line=data[iidx[id]];
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
        auto lid = atomicInc(cnt,maxv);
        data[lid] = res;
        oidx[atomicInc(wp, maxv)] = lid;
    }
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDAINLINE void drawPoint(Line<Out> line, float w, Uniform uni, FrameBuffer& frameBuffer) {
    auto p = line.pa*w + line.pb*(1.0f - w);
    if (p.z >= 0.0f & p.z <= 1.0f) {
        auto fo = line.oa*w + line.ob*(1.0f - w);
        fs(p, p.z, fo, uni, frameBuffer);
    }
}

//1...512
template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void drawMicroL(READONLY(Line<Out>) info, 
        READONLY(unsigned int) idx,READONLY(Uniform) uniform, 
        FrameBuffer* frameBuffer, float len) {
    auto line = info[idx[blockIdx.x]];
    auto w = threadIdx.x / len;
    drawPoint<Out,Uniform,FrameBuffer,fs>(line, w, *uniform, *frameBuffer);
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> ds, FSF<Out, Uniform, FrameBuffer> fs>
    CALLABLE void renderLineGPU(const Line<Out>* info,
        const Uniform* uniform, FrameBuffer* frameBuffer,
        unsigned int* cnt,unsigned int* idx,unsigned int lsiz,vec2 fsize,float near,float far) {
    constexpr auto block = 64U;

    if (cnt[10]) cutLines<Out><<<calcSize(cnt[10],block),block>>>
        (cnt[10],&cnt[11],&cnt[9],info, idx + lsiz*10, idx+lsiz*9, fsize, 1 << 9);

    cudaDeviceSynchronize();

    for (auto i = 0; i < 10; ++i)
        if (cnt[i]) {
            dim3 grid(cnt[i]);
            dim3 block(1 << i);
            auto base = idx + i * lsiz;
            drawMicroL<Out, Uniform, FrameBuffer, ds><<<grid,block>>>
                (info,base,uniform, frameBuffer, 1 << i);
        }

    cudaDeviceSynchronize();

    for (auto i = 0; i < 10; ++i)
        if (cnt[i]) {
            dim3 grid(cnt[i]);
            dim3 block(1 << i);
            auto base = idx + i * lsiz;
            drawMicroL<Out, Uniform, FrameBuffer, fs><<<grid, block>>>
                (info,base,uniform, frameBuffer, 1 << i);
        }
}
