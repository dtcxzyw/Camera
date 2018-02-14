#include <ScanLineRenderer/TriangleRasterizer.hpp>

CUDAINLINE void cutTriangle(TriangleRef ref,unsigned int* cnt, TriangleRef* out) {
    const auto by = ref.rect.z;
    for (; ref.rect.x <= ref.rect.y; ref.rect.x += 32.0f) {
        for (ref.rect.z = by; ref.rect.z <= ref.rect.w; ref.rect.z += 32.0f) {
            out[atomicInc(cnt, maxv)] = ref;
        }
    }
}

CALLABLE void emitTriangle(const unsigned int size,unsigned int* cnt,
    READONLY(unsigned int) offset,const TriangleRef* in,TriangleRef* out) {
    const auto id = getID();
    if(id>=size)return;
    const auto ref = in[id];
    if (ref.size== 5)cutTriangle(ref,cnt+4,out+offset[4]);
    else out[offset[ref.size] + atomicInc(cnt+ref.size,maxv)]=ref;
}

CALLABLE void sortTrianglesGPU(unsigned int* cnt, unsigned int* offset,unsigned int* tmp,
    TriangleRef* ref, TriangleRef* out){
    offset[0] = 0;
    for (auto i = 1; i < 5; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
    for (auto i = 0; i < 5; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    run(emitTriangle,block,cnt[6],tmp,offset,ref,out);
    cudaDeviceSynchronize();
    offset[5] = offset[4] + tmp[4];
}

std::pair<MemoryRef<unsigned int>, MemoryRef<TriangleRef>> sortTriangles(CommandBuffer& buffer,
    const MemoryRef<unsigned int>& cnt, const MemoryRef<TriangleRef>& ref){
    auto sortedIdx = buffer.allocBuffer<TriangleRef>(ref.size()*2);
    auto tmp = buffer.allocBuffer<unsigned int>(5);
    auto offset = buffer.allocBuffer<unsigned int>(6);
    buffer.callKernel(sortTrianglesGPU, cnt, offset, tmp, ref, sortedIdx);
    return { offset,sortedIdx };
}
