#pragma once
#include <Core/Common.hpp>
#include <Math/Geometry.hpp>

DEVICEINLINE Point toRaster(const Point pos, const vec2 hsiz) {
    const Vector p{pos};
    const auto invz = 1.0f / p.z;
    return {(1.0f + p.x * invz) * hsiz.x, (1.0f - p.y * invz) * hsiz.y, invz};
}

constexpr auto tileOffset = 1.0f;

DEVICEINLINE int calcTileSize(const uvec4 rect) {
    const auto delta = max(2U, max(rect.y - rect.x, rect.w - rect.z));
    const auto bit = glm::findMSB(delta);
    return min(bit + ((1U << bit) != delta) - 1, 5);
}

DEVICEINLINE float shuffleFloat(const float w, const int laneMask) {
    constexpr auto mask = 0xffffffff;
    return __shfl_xor_sync(mask, w, laneMask);
}

DEVICEINLINE Vector shuffleVector(const Vector w, const int laneMask) {
    return {
        shuffleFloat(w.x, laneMask),
        shuffleFloat(w.y, laneMask),
        shuffleFloat(w.z, laneMask)
    };
}

template <typename Uniform>
using PosConverter = Point(*)(Point cameraPos, const Uniform& uniform);
