#pragma once
#include <Base/Pipeline.hpp>

template<typename Vert, typename Out, typename Uniform>
using VSF = void(*)(Vert in,Uniform uniform, vec4& pos,Out& out);

template<typename Out, typename Uniform, typename FrameBuffer>
using FSF = void(*)(ivec2 uv,float z, Out in, Uniform uniform,
    FrameBuffer& frameBuffer);

CUDA void toNDC(vec4& p,uvec2 size);

template<typename Vert, typename Out, typename Uniform,VSF<Vert, Out, Uniform> vs>
CALLABLE void runVS(unsigned int size,const Vert* ReadOnly in,const Uniform* ReadOnly u,
    std::pair<vec4,Out>* out,uvec2 fsize) {
    auto i = getID();
    if (i >= size)return;
    vs(in[i], *u, out[i].first, out[i].second);
    toNDC(out[i].first, fsize);
}

template<typename Uniform, typename FrameBuffer>
using FSFSF = void(*)(uvec2 NDC, Uniform uniform, FrameBuffer& frameBuffer);

template<typename Uniform, typename FrameBuffer,FSFSF<Uniform,FrameBuffer> fs>
    CALLABLE void runFSFS(unsigned int size,vec4* pos,const Uniform* ReadOnly u,
        FrameBuffer* frameBuffer) {
    auto i = getID();
    if (i >= size)return;
    auto pixel = frameBuffer->size();
    auto x = i%pixel.x, y = i / pixel.x;
    fs({ x,y }, *u, *frameBuffer);
}

