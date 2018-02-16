#pragma once
#include <Base/DispatchSystem.hpp>
#include <Base/Math.hpp>

template<typename Uniform, typename FrameBuffer>
using FSFSF = void(*)(ivec2 NDC, const Uniform& uniform, FrameBuffer frameBuffer);

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform, FrameBuffer> fs>
GLOBAL void runFSFS(READONLY(Uniform) u,
    FrameBuffer frameBuffer, const uvec2 size) {
    const auto x = blockIdx.x*blockDim.x + threadIdx.x;
    const auto y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x<size.x & y<size.y)
        fs(ivec2{ x,y }, *u, frameBuffer);
}

template<typename Uniform, typename FrameBuffer, FSFSF<Uniform, FrameBuffer> fs>
void renderFullScreen(CommandBuffer& buffer, const DataPtr<Uniform>& uniform,
    const Value<FrameBuffer>& frameBuffer, uvec2 size) {
    constexpr auto tileSize = 32U;
    dim3 grid(calcSize(size.x, tileSize), calcSize(size.y, tileSize));
    dim3 block(tileSize, tileSize);
    buffer.runKernelDim(runFSFS<Uniform, FrameBuffer, fs>, grid, block, uniform.get(),
        frameBuffer.get(), size);
}

