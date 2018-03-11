#include <ScanLineRenderer/LineRasterizer.hpp>

CUDAINLINE void cutLine(LineRef ref, unsigned int* cnt, LineRef* out) {
    const auto offset = ref.range.x;
    const auto invdis = 1.0f / ref.len;
    const auto mul = ref.range.y *invdis;
    constexpr auto step = 1024.0f, invstep = 1.0f / step;
    const auto refCnt = ceil(ref.len *invstep);
    auto base = atomicAdd(cnt, refCnt);
    for (auto i = 0; i < refCnt; ++i) {
        const auto x = step * i;
        const auto y = fmin(ref.len, x + step);
        ref.range.x = x * mul + offset;
        ref.range.y = (y - x) *mul;
        out[base++] = ref;
    }
}

GLOBAL void emitLine(const unsigned int size, unsigned int* cnt,
                         READONLY(unsigned int) offset, const LineRef* in, LineRef* out) {
    const auto id = getId();
    if (id >= size)return;
    const auto ref = in[id];
    if (ref.size == 11)cutLine(ref, cnt + 10, out + offset[10]);
    else out[offset[ref.size] + atomicInc(cnt + ref.size, maxv)] = ref;
}

GLOBAL void sortLinesGPU(unsigned int* cnt, unsigned int* offset, unsigned int* tmp,
                             LineRef* ref, LineRef* out) {
    offset[0] = 0;
#pragma unroll
    for (auto i = 1; i < 11; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
#pragma unroll
    for (auto i = 0; i < 11; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    run(emitLine, block, cnt[12], tmp, offset, ref, out);
    cudaDeviceSynchronize();
    offset[11] = offset[10] + tmp[10];
}

std::pair<MemoryRef<unsigned int>, MemoryRef<LineRef>> sortLines(CommandBuffer& buffer,
                                                                 const MemoryRef<unsigned int>& cnt,
                                                                 const MemoryRef<LineRef>& ref) {
    auto sortedIdx = buffer.allocBuffer<LineRef>(ref.size() * 2U+2048U);
    auto tmp = buffer.allocBuffer<unsigned int>(11);
    auto offset = buffer.allocBuffer<unsigned int>(12);
    buffer.callKernel(sortLinesGPU, cnt, offset, tmp, ref, sortedIdx);
    return {offset, sortedIdx};
}
