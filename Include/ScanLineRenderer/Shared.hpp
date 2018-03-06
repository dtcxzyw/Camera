#pragma once
#include <Base/Common.hpp>
#include <Base/Math.hpp>
#include <Base/DispatchSystem.hpp>

CUDAINLINE vec3 toRaster(const vec3 p, const vec2 hsiz) {
    const auto invz = 1.0f / p.z;
    return { (1.0f + p.x * invz)*hsiz.x,(1.0f - p.y*invz)*hsiz.y,invz };
}

constexpr auto tileOffset = 1.0f;

CUDAINLINE int calcTileSize(const uvec4 rect) {
    const float dx = rect.y - rect.x, dy = rect.w - rect.z;
    const auto tsiz = fmax(1.5f, fmax(dx, dy));
    return ceil(fmin(log2f(tsiz) - 1.0f, 4.9f));
}

struct TileRef final {
    unsigned int id;
    unsigned int size;
    uvec4 rect;
};

std::pair<MemoryRef<unsigned int>, MemoryRef<TileRef>> sortTiles(CommandBuffer& buffer,
    const MemoryRef<unsigned int>& cnt, const MemoryRef<TileRef>& ref,size_t refSize,
    unsigned int maxSize);

CUDAINLINE float shuffleFloat(const float w,const int laneMask) {
    constexpr auto mask = 0xffffffff;
    return __shfl_xor_sync(mask, w, laneMask);
}

CUDAINLINE vec3 shuffleVec3(const vec3 w, const int laneMask) {
    return {
        shuffleFloat(w.x,laneMask),
        shuffleFloat(w.y,laneMask),
        shuffleFloat(w.z,laneMask)
    };
}

template <typename Uniform>
using PosConverter = vec3(*)(vec3 cameraPos,const Uniform& uniform);

CUDAINLINE float max3(const float a, const float b, const float c) {
    return fmax(a, fmax(b, c));
}

CUDAINLINE float min3(const float a, const float b, const float c) {
    return fmin(a, fmin(b, c));
}
