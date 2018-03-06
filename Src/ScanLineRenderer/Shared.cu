#include <ScanLineRenderer/Shared.hpp>
#include <Base/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>

CUDAINLINE void cutTile(TileRef ref, unsigned int* cnt, TileRef* out, const unsigned int maxSize) {
    const auto by = ref.rect.z;
    constexpr auto step = 32U;
    const int xcnt = calcBlockSize(ref.rect.y - ref.rect.x, step);
    const int ycnt = calcBlockSize(ref.rect.w - ref.rect.z,step);
    const auto size= xcnt * ycnt;
    auto base = atomicAdd(cnt, size);
    if (base + size > maxSize)return;
    for (; ref.rect.x <= ref.rect.y; ref.rect.x += step) {
        for (ref.rect.z = by; ref.rect.z <= ref.rect.w; ref.rect.z += step) {
            out[base++] = ref;
        }
    }
}

GLOBAL void emitTile(const unsigned int size,unsigned int* cnt,READONLY(unsigned int) offset,
    READONLY(TileRef) in, TileRef* out, const unsigned int maxSize) {
    const auto id = getID();
    if(id>=size)return;
    const auto ref = in[id];
    if (ref.size== 5)cutTile(ref,cnt+5,out+offset[5],maxSize);
    else out[offset[ref.size] + atomicInc(cnt+ref.size,maxv)]=ref;
}

GLOBAL void sortTilesGPU(unsigned int* cnt, unsigned int* offset,unsigned int* tmp,
    TileRef* ref, TileRef* out, const unsigned int maxSize,const unsigned int maxOutSize){
    auto launchSize = cnt[6];
    if (cnt[6] > maxSize)launchSize = maxSize;

    offset[0] = 0;
    for (auto i = 1; i < 6; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
    for (auto i = 0; i < 6; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    run(emitTile, block, launchSize, tmp, offset, ref, out, maxOutSize);
    cudaDeviceSynchronize();
    offset[5] += tmp[5];
}

std::pair<MemoryRef<unsigned int>, MemoryRef<TileRef>> sortTiles(CommandBuffer& buffer,
    const MemoryRef<unsigned int>& cnt, const MemoryRef<TileRef>& ref, const size_t refSize,
    const unsigned int maxSize) {
    auto sortedIdx = buffer.allocBuffer<TileRef>(refSize);
    auto tmp = buffer.allocBuffer<unsigned int>(6);
    auto offset = buffer.allocBuffer<unsigned int>(6);
    const unsigned int maxOutSize=sortedIdx.maxSize();
    buffer.callKernel(sortTilesGPU, cnt, offset, tmp, ref, sortedIdx, maxSize, maxOutSize);
    return { offset,sortedIdx };
}
