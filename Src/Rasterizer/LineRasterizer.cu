#include <Rasterizer/LineRasterizer.hpp>
#include <Core/DeviceFunctions.hpp>

DEVICEINLINE void cutLine(LineRef ref, uint32_t* cnt, LineRef* out) {
    const auto offset = ref.range.x;
    const auto invdis = 1.0f / ref.len;
    const auto mul = ref.range.y * invdis;
    constexpr auto step = 1024.0f, invstep = 1.0f / step;
    const uint32_t refCnt = ceil(ref.len * invstep);
    auto base = deviceAtomicAdd(cnt, refCnt);
    for (auto i = 0; i < refCnt; ++i) {
        const auto x = step * i;
        const auto y = fmin(ref.len, x + step);
        ref.range.x = x * mul + offset;
        ref.range.y = (y - x) * mul;
        out[base++] = ref;
    }
}

GLOBAL void emitLine(const uint32_t size, uint32_t* cnt,
    READONLY(uint32_t) offset, const LineRef* in, LineRef* out) {
    const auto id = getId();
    if (id >= size)return;
    const auto ref = in[id];
    if (ref.size == 11)cutLine(ref, cnt + 10, out + offset[10]);
    else out[offset[ref.size] + deviceAtomicInc(cnt + ref.size, maxv)] = ref;
}

GLOBAL void sortLinesKernel(uint32_t* cnt, uint32_t* offset, uint32_t* tmp,
    LineRef* ref, LineRef* out) {
    offset[0] = 0;
    #pragma unroll
    for (auto i = 1; i < 11; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
    #pragma unroll
    for (auto i = 0; i < 11; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    launchLinear(emitLine, block, cnt[12], tmp, offset, ref, out);
    cudaDeviceSynchronize();
    offset[11] = offset[10] + tmp[10];
}

std::pair<Span<uint32_t>, Span<LineRef>> sortLines(CommandBuffer& buffer,
    const Span<uint32_t>& cnt,
    const Span<LineRef>& ref) {
    auto sortedIdx = buffer.allocBuffer<LineRef>(ref.size() * 2U + 2048U);
    auto tmp = buffer.allocBuffer<uint32_t>(11);
    auto offset = buffer.allocBuffer<uint32_t>(12);
    buffer.callKernel(makeKernelDesc(sortLinesKernel), cnt, offset, tmp, ref, sortedIdx);
    return {offset, sortedIdx};
}
