#pragma once
#include <ScanLineRenderer/ScanLine.hpp>
#include <device_atomic_functions.h>

template<typename Vert>
struct Queue final {
private:
    unsigned int* const mCnt;
    Vert* const mVert;
public:
    Queue(unsigned int* cnt,Vert* ptr):mCnt(cnt),mVert(ptr){}
    template<unsigned int size>
    CUDA void push(Vert* vert) {
        auto id=atomicAdd(mCnt,size);
        for(int i=0;i<size;++i)mVert[id+i] = vert[i];
    }
};

template<typename Vert, typename Uniform>
using GTSF = void(*)(Vert* in, Uniform uniform,Queue<Vert> queue);

template<typename Index, typename Vert, typename Uniform, GTSF<Vert, Uniform> gs>
CALLABLE void GTHelper(unsigned int size, const Vert* ReadOnly vert, Index idx,
    const Uniform* ReadOnly uniform,Queue<Vert> queue) {
    auto id = getID();
    if (id >= size)return;
    Vert in[3];
    in[0] = vert[idx[id][0]], in[1] = vert[idx[id][1]], in[2] = vert[idx[id][2]];
    gs(in, *uniform, queue);
}

template<typename Index,typename Vert,typename Uniform,GTSF<Vert,Uniform> gs>
auto genTriangle(Pipeline& pipeline,DataViewer<Vert> vert,Index idx,const Uniform* uniform
    ,unsigned int outSize=0U) {
    if (outSize == 0U)outSize = idx.size();
    outSize *= 3;
    auto res = allocBuffer<Vert>(outSize);
    auto cnt = allocBuffer<unsigned int>();
    checkError(cudaMemsetAsync(cnt.begin(),0,sizeof(unsigned int),pipeline.getId()));
    pipeline.run(GTHelper<Index, Vert, Uniform, gs>, idx.size(), vert.begin(), idx, uniform
        , Queue<Vert>{cnt.begin(),res.begin()});
    pipeline.sync();
    res.scale(*cnt);
    return res;
}
