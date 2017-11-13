#pragma once
#include <ScanLineRenderer/ScanLine.hpp>
#include <device_atomic_functions.h>

template<typename Vert, unsigned int size>
struct Queue final {
private:
    unsigned int* const mCnt;
    Vert* const mVert;
public:
    Queue(unsigned int* cnt,Vert* ptr):mCnt(cnt),mVert(ptr){}
    CUDA void push(Vert* vert) {
        auto id=atomicAdd(mCnt,size);
        for(int i=0;i<size;++i)mVert[id+i] = vert[i];
    }
};

template<typename Vert, typename Uniform,unsigned int size>
using GSF = void(*)(Vert* in, Uniform uniform,Queue<Vert,size> queue);

template<unsigned int vsiz,typename Index, typename Vert, typename Uniform
    , GSF<Vert, Uniform,vsiz> gs>
CALLABLE void GTHelper(unsigned int size, const Vert* ReadOnly vert, Index idx,
    const Uniform* ReadOnly uniform,Queue<Vert,vsiz> queue) {
    auto id = getID();
    if (id >= size)return;
    Vert in[vsiz];
    for (int i = 0; i < vsiz; ++i)in[i] = vert[idx[id][i]];
    gs(in, *uniform, queue);
}

template<unsigned int size,typename Index,typename Vert,typename Uniform
    ,GSF<Vert,Uniform,size> gs>
auto genPrimitive(Pipeline& pipeline,DataViewer<Vert> vert,Index idx,const Uniform* uniform
    ,unsigned int outSize=0U) {
    if (outSize == 0U)outSize = idx.size();
    outSize *= size;
    auto res = allocBuffer<Vert>(outSize);
    auto cnt = allocBuffer<unsigned int>();
    checkError(cudaMemsetAsync(cnt.begin(),0,sizeof(unsigned int),pipeline.getId()));
    pipeline.run(GTHelper<size,Index, Vert, Uniform, gs>, idx.size(), vert.begin(), idx, uniform
        , Queue<Vert,size>{cnt.begin(),res.begin()});
    pipeline.sync();
    res.scale(*cnt);
    return res;
}
