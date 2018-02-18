#include <ScanLineRenderer/LineRasterizer.hpp>

CUDAINLINE void cutTriangle(LineRef ref, unsigned int* cnt, LineRef* out) {
    const auto rect = ref.rect;
    const auto dis = distance(vec2(rect.x, rect.z), vec2(rect.y, rect.w));
    for(auto x=0.0f;x<=dis;x+=1024.0f) {
        const auto y = fmin(dis,x+1024.0f);
        ref.range.x = x / dis;
        ref.range.y = (y-x) / dis;
        out[atomicInc(cnt,maxv)]=ref;
    }
}

GLOBAL void emitTriangle(const unsigned int size, unsigned int* cnt,
                         READONLY(unsigned int) offset, const LineRef* in, LineRef* out) {
    const auto id = getID();
    if (id >= size)return;
    const auto ref = in[id];
    if (ref.size == 5)cutTriangle(ref, cnt + 10, out + offset[10]);
    else out[offset[ref.size] + atomicInc(cnt + ref.size, maxv)] = ref;
}

GLOBAL void sortTrianglesGPU(unsigned int* cnt, unsigned int* offset, unsigned int* tmp,
                             LineRef* ref, LineRef* out) {
    offset[0] = 0;
#pragma unroll
    for (auto i = 1; i < 11; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
#pragma unroll
    for (auto i = 0; i < 11; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    run(emitTriangle, block, cnt[12], tmp, offset, ref, out);
    cudaDeviceSynchronize();
    offset[11] = offset[10] + tmp[10];
}

std::pair<MemoryRef<unsigned int>, MemoryRef<LineRef>> sortLines(CommandBuffer& buffer,
                                                                 const MemoryRef<unsigned int>& cnt,
                                                                 const MemoryRef<LineRef>& ref) {
    auto sortedIdx = buffer.allocBuffer<LineRef>(ref.size() * 2);
    auto tmp = buffer.allocBuffer<unsigned int>(11);
    auto offset = buffer.allocBuffer<unsigned int>(12);
    buffer.callKernel(sortTrianglesGPU, cnt, offset, tmp, ref, sortedIdx);
    return {offset, sortedIdx};
}
