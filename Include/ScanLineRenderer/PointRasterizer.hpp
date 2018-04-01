#pragma once
#include <ScanLineRenderer/Vertex.hpp>
#include <ScanLineRenderer/Shared.hpp>

template <typename Out, typename Uniform, typename FrameBuffer>
using FSFP = void(*)(unsigned int id, ivec2 uv, float z, const Out& in,
                     const Uniform& uniform, FrameBuffer& frameBuffer);

template <typename Out, typename Uniform, typename FrameBuffer, 
    PosConverter<Uniform> toPos, FSFP<Out, Uniform, FrameBuffer> fs>
GLOBAL void drawPointHelperKernel(const unsigned int size, READONLY(VertexInfo<Out>) vert,
                               READONLY(Uniform) uniform, FrameBuffer* frameBuffer, const vec4 scissor,
                               const float near, const float invnf, const vec2 hfsize) {
    const auto id = getId();
    if (id >= size)return;
    auto p = vert[id];
    p.pos = toPos(p.pos, *uniform);
    auto nz = (p.pos.z - near) * invnf;
    p.pos = toRaster(p.pos, hfsize);
    if (scissor.x <= p.pos.x & p.pos.x <= scissor.y & scissor.z <= p.pos.y & p.pos.y <= scissor.w
        & 0.0f <= nz & nz <= 1.0f) {
        fs(id, {p.pos.x, p.pos.y}, nz, p.out, *uniform, *frameBuffer);
    }
}

template <typename Out, typename Uniform, typename FrameBuffer,PosConverter<Uniform> toPos>
void drawPointHelper(CommandBuffer&, const DataPtr<VertexInfo<Out>>&,
                     const DataPtr<Uniform>&, const DataPtr<FrameBuffer>&,const vec4,
                     const float, const float, const vec2) {}

template <typename Out, typename Uniform, typename FrameBuffer, PosConverter<Uniform> toPos,
    FSFP<Out, Uniform, FrameBuffer> first, FSFP<Out, Uniform, FrameBuffer>... then>
void drawPointHelper(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
                     const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer,const vec4 scissor,
                     const float near, const float invnf, const vec2 hfsize) {
    buffer.launchKernelLinear(drawPointHelperKernel<Out, Uniform, FrameBuffer,toPos, first>, vert.size(), vert.get(),
        uniform.get(), frameBuffer.get(), scissor, near, invnf, hfsize);
    drawPointHelper<Out, Uniform, FrameBuffer,toPos, then...>(buffer, vert, uniform, frameBuffer, scissor,
        near, invnf, hfsize);
}

template <typename Out, typename Uniform, typename FrameBuffer, 
    PosConverter<Uniform> toPos, FSFP<Out, Uniform, FrameBuffer>... fs>
void renderPoints(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
                  const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer,
                  const uvec2 size, const float near, const float far,vec4 scissor) {
    scissor = { fmax(0.5f,scissor.x),fmin(size.x - 0.5f,scissor.y),
        fmax(0.5f,scissor.z),fmin(size.y - 0.5f,scissor.w) };
    const auto hfsize = static_cast<vec2>(size) * 0.5f;
    const auto invnf = 1.0f / (far - near);
    drawPointHelper<Out, Uniform, FrameBuffer,toPos, fs...>(buffer, vert, uniform, frameBuffer, 
        scissor,near, invnf, hfsize);
}
