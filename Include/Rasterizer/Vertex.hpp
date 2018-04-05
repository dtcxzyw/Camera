#pragma once
#include <Base/Common.hpp>
#include  <Base/Math.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/Cache.hpp>

template<typename Vert, typename Out, typename Uniform>
using VertShader = void(*)(Vert in, const Uniform& uniform, vec3& pos, Out& out);

template<typename Out>
struct STRUCT_ALIGN VertexInfo {
    vec3 pos;//in camera space
    Out out;
};

template<typename Out>
CUDAINLINE VertexInfo<Out> lerpZ(VertexInfo<Out> a, VertexInfo<Out> b, float z) {
    auto u = (z - b.pos.z) / (a.pos.z - b.pos.z), v = 1.0f - u;
    VertexInfo<Out> res;
    res.pos = { a.pos.x*u + b.pos.x*v,a.pos.y*u + b.pos.y*v,z };
    res.out = a.out*u + b.out*v;
    return res;
}

template<typename Vert, typename Out, typename Uniform, VertShader<Vert, Out, Uniform> vs>
GLOBAL void calcVertexKernel(const unsigned int size, READONLY(Vert) in,
    READONLY(Uniform) u, VertexInfo<Out>* res) {
    const auto id = getId();
    if (id >= size)return;
    auto& vert = res[id];
    vs(in[id], *u, vert.pos, vert.out);
}

template<typename Out,typename Judge>
using VertexCache = CachedMemoryHolder<VertexInfo<Out>, Judge>;

template<typename Vert, typename Out, typename Uniform, VertShader<Vert, Out, Uniform> Func,
    typename Judge = EmptyJudge>
DataPtr<VertexInfo<Out>> calcVertex(CommandBuffer& buffer, const DataPtr<Vert>& vert,
    const DataPtr<Uniform>& uniform,
    CacheRef<DataViewer<VertexInfo<Out>>, Judge> cache = {}) {
    if (cache && cache.vaild())return cache.getValue();
    auto vertex = buffer.allocBuffer<VertexInfo<Out>>(vert.size(), 
        updateMemory(cache.getRef(), cache.getJudge()));
    buffer.launchKernelLinear(calcVertexKernel<Vert, Out, Uniform, Func>, vert.size(), vert.get(),
        uniform.get(), vertex);
    return vertex;
}
