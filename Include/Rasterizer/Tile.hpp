#pragma once
#include <Core/CommandBuffer.hpp>
#include <Core/DeviceFunctions.hpp>

struct TileRef final {
    uint32_t id;
    uint32_t size;
    uvec4 rect;
};

template <typename Uniform, typename RefData>
using TileClipShader = bool(*)(const TileRef& ref, const Uniform& uniform,READONLY(RefData) data);

DEVICEINLINE bool emptyTileClipShader(const TileRef&, const Empty&, const unsigned char*) {
    return true;
}

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
DEVICEINLINE void cutTile(TileRef ref, uint32_t* cnt, TileRef* out, const uint32_t maxSize,
    const Uniform* uni, READONLY(RefData) data) {
    constexpr auto step = 32U;
    const auto by = ref.rect.z;
    for (; ref.rect.x <= ref.rect.y; ref.rect.x += step) {
        for (ref.rect.z = by; ref.rect.z <= ref.rect.w; ref.rect.z += step) {
            if (Func(ref, *uni, data)) {
                const auto id = deviceAtomicInc(cnt, maxv);
                if (id >= maxSize)return;
                out[id] = ref;
            }
        }
    }
}

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
GLOBAL void emitTile(const uint32_t size, uint32_t* cnt, READONLY(uint32_t) offset,
    READONLY(TileRef) in, TileRef* out, const uint32_t maxSize,
    const Uniform* uni, READONLY(RefData) data) {
    const auto id = getId();
    if (id >= size)return;
    const auto ref = in[id];
    if (ref.size == 5)cutTile<Uniform, RefData, Func>(ref, cnt + 5, out + offset[5], maxSize, uni, data);
    else out[offset[ref.size] + deviceAtomicInc(cnt + ref.size, maxv)] = ref;
}

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
GLOBAL void sortTilesKernel(uint32_t* cnt, uint32_t* offset, uint32_t* tmp,
    TileRef* ref, TileRef* out, const uint32_t maxSize, const uint32_t maxOutSize,
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
    Span<uint32_t> cnt;
    Span<TileRef> array;

    TileProcessingResult(const Span<unsigned>& cnt, const Span<TileRef>& array)
        : cnt(cnt), array(array) {}
};

template <typename Uniform, typename RefData, TileClipShader<Uniform, RefData> Func>
TileProcessingResult sortTiles(CommandBuffer& buffer,
    const Span<uint32_t>& cnt, const Span<TileRef>& ref, const size_t refSize,
    const uint32_t maxSize, const Span<Uniform>& uni, const Span<RefData>& data) {
    auto sortedIdx = buffer.allocBuffer<TileRef>(refSize);
    auto tmp = buffer.allocBuffer<uint32_t>(6);
    auto offset = buffer.allocBuffer<uint32_t>(6);
    const uint32_t maxOutSize = sortedIdx.maxSize();
    buffer.callKernel(makeKernelDesc(sortTilesKernel<Uniform, RefData, Func>), cnt, offset, tmp, ref,
        sortedIdx, maxSize, maxOutSize, uni, data);
    return {offset, sortedIdx};
}
