#pragma once
#include <Base/Pipeline.hpp>

template<typename Vert, typename Out, typename Uniform>
using VSF = void(*)(Vert in,Uniform uniform, vec4& pos,Out& out);

template<typename Out, typename Uniform, typename FrameBuffer>
using FSF = void(*)(ivec2 uv,unsigned int z, Out in, Uniform uniform,
    FrameBuffer& frameBuffer);

inline CUDA vec3 toNDC(vec4 p, vec2 size) {
    return { (0.5f + p.x / p.w*0.5f)*size.x,
        (0.5f - p.y / p.w*0.5f)*size.y,
        p.z/p.w + epsilon<float>() };
}

inline CUDA int checkPoint(vec3 p,vec2 size) {
    return (p.x >= 0.0f)
        | (p.x < size.x) << 1
        | (p.y >= 0.0f) << 2
        | (p.y < size.y) << 3
        | (p.z >= 0.0f) << 4
        | (p.z <= 1.0f) << 5;
}

template<typename Out>
struct VertexInfo {
    vec3 pos;
    int flag;
    Out out;
};

template<typename Vert, typename Out, typename Uniform,VSF<Vert, Out, Uniform> vs>
CALLABLE void runVS(unsigned int size,const Vert* ReadOnly in,const Uniform* ReadOnly u,
    VertexInfo<Out>* res,vec2 fsize) {
    auto i = getID();
    if (i >= size)return;
    vec4 pos;
    vs(in[i], *u, pos, res[i].out);
    res[i].pos=toNDC(pos, fsize);
    res[i].flag = checkPoint(res[i].pos,fsize);
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
