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
using FSFSF = void(*)(ivec2 NDC, Uniform uniform, FrameBuffer frameBuffer);

template<typename Uniform, typename FrameBuffer,FSFSF<Uniform,FrameBuffer> fs>
    CALLABLE void runFSFS(unsigned int size,const Uniform* ReadOnly u,
        FrameBuffer frameBuffer,unsigned px) {
    auto i = getID();
    if (i >= size)return;
    fs(ivec2{ i%px,i / px }, *u, frameBuffer);
}

