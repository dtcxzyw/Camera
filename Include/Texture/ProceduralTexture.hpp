#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

CUDAINLINE vec3 uv(const vec2 uv) {
    return { uv.x - floor(uv.x),uv.y - floor(uv.y),0.0f };
}

CUDAINLINE bool checkerBoard(const vec2 uv) {
    return (uv.x - floor(uv.x) > 0.5f) ^ (uv.y - floor(uv.y) < 0.5f);
}

CUDAINLINE bool checkerBoard3D(const vec3 uvw) {
    const int val = floor(uvw.x) + floor(uvw.y) + floor(uvw.z);
    return val & 1;
}

