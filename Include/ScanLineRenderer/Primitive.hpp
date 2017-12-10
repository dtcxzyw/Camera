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

template<unsigned int inv,unsigned int outv,typename Index, typename Vert, typename Uniform
    , GSF<Vert, Uniform,outv> gs>
CALLABLE void GTHelper(unsigned int size, const Vert* ReadOnlyCache vert, Index idx,
    const Uniform* ReadOnlyCache uniform,Queue<Vert,outv> queue) {
    auto id = getID();
    if (id >= size)return;
    Vert in[inv];
    for (int i = 0; i < inv; ++i)in[i] = vert[idx[id][i]];
    gs(in, *uniform, queue);
}

template<unsigned int inv,unsigned int outv,typename Index,typename Vert,typename Uniform
    ,GSF<Vert,Uniform,outv> gs>
auto genPrimitive(CommandBuffer& buffer,const MemoryRef<Vert>& vert,Index idx
    ,const MemoryRef<Uniform>& uniform,unsigned int outSize=0U) {
    if (outSize == 0U)outSize = idx.size();
    outSize *= outv;
    auto res = buffer.allocBuffer<Vert>(outSize);
    auto cnt = buffer.allocBuffer<unsigned int>();
    buffer.memset(cnt);
    buffer.runKernelLinear(GTHelper<inv,outv,Index, Vert, Uniform, gs>, idx.size(), vert.begin(), idx
        , uniform,buffer.makeLazyConstructor<Queue<Vert,outv>>(cnt,res));
    return std::make_pair(res,cnt);
}

template<unsigned int inv, unsigned int outv, typename Index, typename Vert, typename Uniform
    , GSF<Vert, Uniform, outv> gs>
    auto genPrimitive(Stream& stream, DataViewer<Vert> vert, Index idx, const Uniform* uniform
        , unsigned int outSize = 0U) {
    if (outSize == 0U)outSize = idx.size();
    outSize *= outv;
    auto res = allocBuffer<Vert>(outSize);
    auto cnt = allocBuffer<unsigned int>();
    checkError(cudaMemsetAsync(cnt.begin(), 0, sizeof(unsigned int), stream.getID()));
    stream.run(GTHelper<inv, outv, Index, Vert, Uniform, gs>, idx.size(), vert.begin(), idx, uniform
        , Queue<Vert, outv>{cnt.begin(), res.begin()});
    stream.sync();
    res.scale(*cnt);
    return res;
}
