#pragma once
#include <Core/Queue.hpp>

template <typename Vert, typename Uniform>
using GeometryShader = void(*)(Vert* in, const Uniform& uniform, QueueRef<Vert> queue);

template <uint32_t In, typename Index, typename Vert, typename Uniform,
    GeometryShader<Vert, Uniform> Func>
GLOBAL void genPrimitiveKernel(const uint32_t size,READONLY(Vert) vert, Index idx,
    READONLY(Uniform) uniform, QueueRef<Vert> queue) {
    const auto id = getId();
    if (id >= size)return;
    Vert in[In];
    for (auto i = 0; i < In; ++i)in[i] = vert[idx[id][i]];
    Func(in, *uniform, queue);
}

template <uint32_t In, uint32_t Out, typename Index, typename Vert, typename Uniform
    , GeometryShader<Vert, Uniform> Func>
auto genPrimitive(CommandBuffer& buffer, const Span<Vert>& vert, Index idx
    , const Span<Uniform>& uniform, uint32_t outSize = 0U) {
    if (outSize == 0U)outSize = idx.size();
    outSize *= Out;
    Queue<Vert> out(buffer, outSize);
    buffer.launchKernelLinear(
        makeKernelDesc(genPrimitiveKernel<In, Index, Vert, Uniform, Func>), idx.size(),
        vert, idx, uniform, out.get(buffer));
    return out;
}
