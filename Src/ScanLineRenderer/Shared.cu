#include <ScanLineRenderer/Shared.hpp>
#include <Base/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>

CUDAINLINE void cutTile(TileRef ref, unsigned int* cnt, TileRef* out) {
    const auto bx=ref.rect.x,by = ref.rect.z;
    constexpr auto step = 32.0f,invstep=1.0f/step;
    const auto xcnt = ceil((ref.rect.y - ref.rect.x) *invstep);
    const auto ycnt = ceil((ref.rect.w - ref.rect.z)*invstep);
    auto base = atomicAdd(cnt, xcnt*ycnt);
    for (auto x = 0; x < xcnt; ++x) {
        for (auto y = 0; y < ycnt; ++y) {
            ref.rect.x = bx + step * x;
            ref.rect.z = by + step * y;
            out[base++] = ref;
        }
    }
}

GLOBAL void emitTile(const unsigned int size,unsigned int* cnt,
    READONLY(unsigned int) offset,const TileRef* in,TileRef* out) {
    const auto id = getID();
    if(id>=size)return;
    const auto ref = in[id];
    if (ref.size== 5)cutTile(ref,cnt+4,out+offset[4]);
    else out[offset[ref.size] + atomicInc(cnt+ref.size,maxv)]=ref;
}

GLOBAL void sortTilesGPU(unsigned int* cnt, unsigned int* offset,unsigned int* tmp,
    TileRef* ref, TileRef* out){
    offset[0] = 0;
    for (auto i = 1; i < 5; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
    for (auto i = 0; i < 5; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    run(emitTile,block,cnt[6],tmp,offset,ref,out);
    cudaDeviceSynchronize();
    offset[5] = offset[4] + tmp[4];
}

std::pair<MemoryRef<unsigned int>, MemoryRef<TileRef>> sortTiles(CommandBuffer& buffer,
    const MemoryRef<unsigned int>& cnt, const MemoryRef<TileRef>& ref, const size_t refSize){
    auto sortedIdx = buffer.allocBuffer<TileRef>(refSize);
    auto tmp = buffer.allocBuffer<unsigned int>(5);
    auto offset = buffer.allocBuffer<unsigned int>(6);
    buffer.callKernel(sortTilesGPU, cnt, offset, tmp, ref, sortedIdx);
    return { offset,sortedIdx };
}
