#pragma once
#include <Base/Pipeline.hpp>

template<typename Vert, typename Out, typename Uniform>
using VSF = void(*)(Vert in,Uniform uniform, Out& out, vec4& pos);

template<typename Out, typename Uniform, typename FrameBuffer>
using FSF = void(*)(ivec2 uv,float z, Out in, Uniform uniform,
    FrameBuffer& frameBuffer);

CUDA void toNDC(vec4& p,uvec2 size);

template<typename Vert, typename Out, typename Uniform,typename FrameBuffer,
    VSF<Vert, Out, Uniform> vs>
CALLABLE void runVS(unsigned int size,const Vert* ReadOnly in,const Uniform* ReadOnly u,
    Out* out,vec4* pos,const FrameBuffer* ReadOnly frameBuffer) {
    auto i = getID();
    if (i >= size)return;
    vs(in[i], *u, out[i], pos[i]);
    toNDC(pos[i],frameBuffer->size());
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

