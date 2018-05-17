#pragma once
#include <Math/Geometry.hpp>
#include <Spectrum/SpectrumConfig.hpp>

DEVICEINLINE Spectrum uv(const vec2 uv) {
    return Spectrum{ RGB{ uv.x - floor(uv.x),uv.y - floor(uv.y),0.0f } };
}

//TODO: Anti-aliasing
DEVICEINLINE Spectrum checkerBoard(const vec2 uv, const Spectrum& a, const Spectrum& b) {
    return (uv.x - floor(uv.x) > 0.5f) ^ (uv.y - floor(uv.y) < 0.5f) ? a : b;
}

DEVICEINLINE Spectrum checkerBoard3D(const Vector& uvw, const Spectrum&a, const Spectrum& b) {
    const int val = floor(uvw.x) + floor(uvw.y) + floor(uvw.z);
    return val & 1 ? a : b;
}
