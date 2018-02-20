#pragma once
#include <ScanLineRenderer/Vertex.hpp>

template<typename Out, typename Uniform, typename FrameBuffer>
using FSFP = void(*)(unsigned int id, ivec2 uv, float z, const Out& in,
    const Uniform& uniform, FrameBuffer& frameBuffer);

template< typename Out, typename Uniform, typename FrameBuffer,
    FSFP<Out, Uniform, FrameBuffer> fs>
    GLOBAL void drawPointHelperGPU(const unsigned int size, READONLY(VertexInfo<Out>) vert,
        READONLY(Uniform) uniform, FrameBuffer* frameBuffer, const vec2 fsize,
        const float near, const float invnf, const vec2 hfsize) {
    const auto id = getID();
    if (id >= size)return;
    auto p = vert[id];
    auto nz = (p.pos.z - near)*invnf;
    p.pos = toRaster(p.pos, hfsize);
    if (0.5f <= p.pos.x & p.pos.x <= fsize.x & 0.5f <= p.pos.y & p.pos.y <= fsize.y
        & 0.0f <= nz & nz <= 1.0f) {
        fs(id, { p.pos.x,p.pos.y }, nz, p.out, *uniform, *frameBuffer);
    }
}

template< typename Out, typename Uniform, typename FrameBuffer>
void drawPointHelper(CommandBuffer&, const DataPtr<VertexInfo<Out>>&,
    const DataPtr<Uniform>&, const DataPtr<FrameBuffer>&, vec2,
    float, float, vec2) {}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSFP<Out, Uniform, FrameBuffer> first, FSFP<Out, Uniform, FrameBuffer>... then>
    void drawPointHelper(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, const DataPtr<FrameBuffer>& frameBuffer, vec2 fsize,
        float near, float invnf, vec2 hfsize) {
    buffer.runKernelLinear(drawPointHelperGPU<Out, Uniform, FrameBuffer, first>, vert.size(), vert,
        uniform, frameBuffer, fsize, near, invnf, hfsize);
    drawPointHelper<Out, Uniform, FrameBuffer, then...>(buffer, vert, uniform, frameBuffer, fsize,
        near, invnf, hfsize);
}

template< typename Out, typename Uniform, typename FrameBuffer,
    FSFP<Out, Uniform, FrameBuffer>... fs>
    void renderPoints(CommandBuffer& buffer, const DataPtr<VertexInfo<Out>>& vert,
        const DataPtr<Uniform>& uniform, FrameBuffer* frameBuffer, const uvec2 size, float near, float far) {
    const auto fsize = static_cast<vec2>(size) - vec2{ 0.5f };
    auto hfsize = static_cast<vec2>(size) * 0.5f;
    auto invnf = 1.0f / (far - near);
    drawPointHelper<Out, Uniform, FrameBuffer, fs...>(buffer, vert, uniform, frameBuffer, fsize,
        near, invnf, hfsize);
}


