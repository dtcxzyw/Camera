#pragma once
#include <Core/Common.hpp>
#include <Core/Math.hpp>

CUDAINLINE vec3 toRaster(const vec3 p, const vec2 hsiz) {
    const auto invz = 1.0f / p.z;
    return { (1.0f + p.x * invz)*hsiz.x,(1.0f - p.y*invz)*hsiz.y,invz };
}

constexpr auto tileOffset = 1.0f;

CUDAINLINE int calcTileSize(const uvec4 rect) {
    const auto delta = max(2U, max(rect.y - rect.x, rect.w - rect.z));
    const auto bit = findMSB(delta);
    return min(bit + ((1U << bit) != delta) - 1, 5);
}

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
