#pragma once
#include <Core/CommandBuffer.hpp>
#include <Math/Math.hpp>

template <typename Uniform, typename FrameBuffer>
using FullScreenShader = void(*)(ivec2 NDC, const Uniform& uniform, FrameBuffer frameBuffer);

template <typename Uniform, typename FrameBuffer,
    FullScreenShader<Uniform, FrameBuffer> FragShader>
GLOBAL void renderFullScreenKernel(READONLY(Uniform) u,
    FrameBuffer frameBuffer, const uvec2 size) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < size.x & y < size.y)
        FragShader(ivec2{x, y}, *u, frameBuffer);
}

template <typename Uniform, typename FrameBuffer,
    FullScreenShader<Uniform, FrameBuffer> FragShader>
void renderFullScreen(CommandBuffer& buffer, const Span<Uniform>& uniform,
    const Value<FrameBuffer>& frameBuffer, uvec2 size) {
    constexpr auto tileSize = 32U;
    dim3 grid(calcBlockSize(size.x, tileSize), calcBlockSize(size.y, tileSize));
    dim3 block(tileSize, tileSize);
    buffer.launchKernelDim(makeKernelDesc(renderFullScreenKernel<Uniform, FrameBuffer, FragShader>),
        grid, block, uniform, frameBuffer.get(), size);
}
