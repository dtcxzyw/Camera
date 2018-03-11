#pragma once
#include <Base/Common.hpp>
#include  <Base/Math.hpp>
#include <Base/DispatchSystem.hpp>

template<typename Vert, typename Out, typename Uniform>
using VSF = void(*)(Vert in, const Uniform& uniform, vec3& pos, Out& out);

template<typename Out>
struct VertexInfo {
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

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
GLOBAL void runVS(const unsigned int size, READONLY(Vert) in,
    READONLY(Uniform) u, VertexInfo<Out>* res) {
    const auto id = getId();
    if (id >= size)return;
    auto& vert = res[id];
    vs(in[id], *u, vert.pos, vert.out);
}

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
auto calcVertex(CommandBuffer& buffer, const DataPtr<Vert>& vert, const DataPtr<Uniform>& uniform) {
    auto vertex = buffer.allocBuffer<VertexInfo<Out>>(vert.size());
    buffer.runKernelLinear(runVS<Vert, Out, Uniform, vs>, vert.size(),vert.get(), uniform.get(), vertex);
    return vertex;
}
