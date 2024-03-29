#pragma once
#include <Core/Common.hpp>
#include <Math/Geometry.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Cache.hpp>

template <typename Vert, typename Out, typename Uniform>
using VertShader = void(*)(Vert in, const Uniform& uniform, Point& pos, Out& out);

template <typename Out>
struct STRUCT_ALIGN VertexInfo {
    Point pos; //in camera space
    Out out;
};

template <typename Out>
DEVICEINLINE VertexInfo<Out> lerpZ(VertexInfo<Out> a, VertexInfo<Out> b, float z) {
    auto u = (z - b.pos.z) / (a.pos.z - b.pos.z), v = 1.0f - u;
    VertexInfo<Out> res;
    res.pos = {a.pos.x * u + b.pos.x * v, a.pos.y * u + b.pos.y * v, z};
    res.out = a.out * u + b.out * v;
    return res;
}

template <typename Vert, typename Out, typename Uniform, VertShader<Vert, Out, Uniform> vs>
GLOBAL void calcVertexKernel(const uint32_t size, READONLY(Vert) in,
    READONLY(Uniform) u, VertexInfo<Out>* res) {
    const auto id = getId();
    if (id >= size)return;
    auto& vert = res[id];
    vs(in[id], *u, vert.pos, vert.out);
}

template <typename Out, typename Judge>
using VertexCache = CachedMemoryHolder<VertexInfo<Out>, Judge>;

template <typename Vert, typename Out, typename Uniform, VertShader<Vert, Out, Uniform> Func,
    typename Judge = EmptyJudge>
Span<VertexInfo<Out>> calcVertex(CommandBuffer& buffer, const Span<Vert>& vert,
    const Span<Uniform>& uniform,
    CacheRef<MemorySpan<VertexInfo<Out>>, Judge> cache = {}) {
    if (cache && cache.vaild())return buffer.useAllocated(cache.getValue());
    cache.reset();
    auto vertex = buffer.allocBuffer<VertexInfo<Out>>(vert.size(),
        updateMemory(cache.getRef(), cache.getJudge()));
    buffer.launchKernelLinear(makeKernelDesc(calcVertexKernel<Vert, Out, Uniform, Func>), vert.size(), vert,
        uniform, vertex);
    return vertex;
}
