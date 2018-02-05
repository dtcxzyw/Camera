#include <ScanLineRenderer/TriangleRasterizer.hpp>

CUDAINLINE void cutTriangle(TriangleRef ref,unsigned int* cnt, TriangleRef* out) {
    const auto bx = ref.rect[0], by = ref.rect[2];
    for (ref.rect[0] = bx; ref.rect[0] <= ref.rect[1]; ref.rect[0] += 32.0f) {
        for (ref.rect[2] = by; ref.rect[2] <= ref.rect[3]; ref.rect[2] += 32.0f) {
            out[atomicInc(cnt, maxv)] = ref;
        }
    }
}

CALLABLE void emitTriangle(const unsigned int size,unsigned int* cnt,
    READONLY(unsigned int) offset,const TriangleRef* in,TriangleRef* out) {
    const auto id = getID();
    if(id>=size)return;
    const auto ref = in[id];
    if (ref.size== 6)cutTriangle(ref,cnt+5,out+offset[5]);
    else out[offset[ref.size] + atomicInc(cnt+ref.size,maxv)]=ref;
}

CALLABLE void sortTrianglesGPU(unsigned int* cnt, unsigned int* offset,unsigned int* tmp,
    TriangleRef* ref, TriangleRef* out){
    offset[0] = 0;
    for (auto i = 1; i < 6; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
    for (auto i = 0; i < 6; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    run(emitTriangle,block,cnt[7],tmp,offset,ref,out);
    cudaDeviceSynchronize();
    offset[6] = offset[5] + tmp[5];
}

std::pair<MemoryRef<unsigned int>, MemoryRef<TriangleRef>> sortTriangles(CommandBuffer& buffer,
    const MemoryRef<unsigned int>& cnt, const MemoryRef<TriangleRef>& ref){
    auto sortedIdx = buffer.allocBuffer<TriangleRef>(ref.size()*2);
    auto tmp = buffer.allocBuffer<unsigned int>(6);
    auto offset = buffer.allocBuffer<unsigned int>(7);
    buffer.callKernel(sortTrianglesGPU, cnt, offset, tmp, ref, sortedIdx);
    return { offset,sortedIdx };
}
