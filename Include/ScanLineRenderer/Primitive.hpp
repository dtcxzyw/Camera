#pragma once
#include <Base/Queue.hpp>

template<typename Vert, typename Uniform>
using GSF = void(*)(Vert* in,const Uniform& uniform,QueueGPU<Vert> queue);

template<unsigned int inv,typename Index, typename Vert, typename Uniform,
    GSF<Vert, Uniform> gs>
GLOBAL void GTHelper(const unsigned int size,READONLY(Vert) vert, Index idx,
    READONLY(Uniform) uniform,QueueGPU<Vert> queue) {
    const auto id = getId();
    if (id >= size)return;
    Vert in[inv];
    for (auto i = 0; i < inv; ++i)in[i] = vert[idx[id][i]];
    gs(in, *uniform, queue);
}

template<unsigned int inv,unsigned int outv,typename Index,typename Vert,typename Uniform
    ,GSF<Vert,Uniform> gs>
auto genPrimitive(CommandBuffer& buffer,const DataPtr<Vert>& vert,Index idx
    ,const DataPtr<Uniform>& uniform,unsigned int outSize=0U) {
    if (outSize == 0U)outSize = idx.size();
    outSize *= outv;
    Queue<Vert> out(buffer,outSize);
    buffer.runKernelLinear(GTHelper<inv,Index, Vert, Uniform, gs>, idx.size(), vert.begin(), idx, 
        uniform,out.get(buffer));
    return out;
}
