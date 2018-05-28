#pragma once
#include <Core/CommandBuffer.hpp>
#include <Core/DataSet.hpp>
#include <Core/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Core/CompileEnd.hpp>

struct TileRef final {
    unsigned int id;
    unsigned int size;
    uvec4 rect;
};

template <typename Uniform, typename RefData>
using TileClipShader = bool(*)(const TileRef& ref, const Uniform& uniform,READONLY(RefData) data);

DEVICEINLINE bool emptyTileClipShader(const TileRef&, const Empty&, const unsigned char*) {
    return true;
}

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
DEVICEINLINE void cutTile(TileRef ref, unsigned int* cnt, TileRef* out, const unsigned int maxSize,
    const Uniform* uni, READONLY(RefData) data) {
    constexpr auto step = 32U;
    const auto by = ref.rect.z;
    for (; ref.rect.x <= ref.rect.y; ref.rect.x += step) {
        for (ref.rect.z = by; ref.rect.z <= ref.rect.w; ref.rect.z += step) {
            if (Func(ref, *uni, data)) {
                const auto id = atomicInc(cnt, maxv);
                if (id >= maxSize)return;
                out[id] = ref;
            }
        }
    }
}

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
GLOBAL void emitTile(const unsigned int size, unsigned int* cnt, READONLY(unsigned int) offset,
    READONLY(TileRef) in, TileRef* out, const unsigned int maxSize,
    const Uniform* uni, READONLY(RefData) data) {
    const auto id = getId();
    if (id >= size)return;
    const auto ref = in[id];
    if (ref.size == 5)cutTile<Uniform, RefData, Func>(ref, cnt + 5, out + offset[5], maxSize, uni, data);
    else out[offset[ref.size] + atomicInc(cnt + ref.size, maxv)] = ref;
}

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
GLOBAL void sortTilesKernel(unsigned int* cnt, unsigned int* offset, unsigned int* tmp,
    TileRef* ref, TileRef* out, const unsigned int maxSize, const unsigned int maxOutSize,
    const Uniform* uni, READONLY(RefData) data) {
    auto launchSize = cnt[6];
    if (launchSize > maxSize)launchSize = maxSize;
    offset[0] = 0;
    for (auto i = 1; i < 6; ++i)offset[i] = offset[i - 1] + cnt[i - 1];
    for (auto i = 0; i < 6; ++i)tmp[i] = 0;
    constexpr auto block = 1024U;
    launchLinear(emitTile<Uniform, RefData, Func>, block, launchSize, tmp, offset, ref, out,
        maxOutSize - offset[5], uni, data);
    cudaDeviceSynchronize();
    offset[5] += tmp[5];
    if (offset[5] > maxOutSize)offset[5] = maxOutSize;
}

struct TileProcessingResult final {
    Span<unsigned int> cnt;
    Span<TileRef> array;

    TileProcessingResult(const Span<unsigned>& cnt, const Span<TileRef>& array)
        : cnt(cnt), array(array) {}
};

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
TileProcessingResult sortTiles(CommandBuffer& buffer,
    const Span<unsigned int>& cnt, const Span<TileRef>& ref, const size_t refSize,
    const unsigned int maxSize, const Span<Uniform>& uni, const Span<RefData>& data) {
    auto sortedIdx = buffer.allocBuffer<TileRef>(refSize);
    auto tmp = buffer.allocBuffer<unsigned int>(6);
    auto offset = buffer.allocBuffer<unsigned int>(6);
    const unsigned int maxOutSize = sortedIdx.maxSize();
    buffer.callKernel(makeKernelDesc(sortTilesKernel<Uniform, RefData, Func>), cnt, offset, tmp, ref,
        sortedIdx, maxSize, maxOutSize, uni, data);
    return {offset, sortedIdx};
}
