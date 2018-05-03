#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <Core/CompileBegin.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtx/spline.hpp>
#include <Core/CompileEnd.hpp>

CUDAINLINE bool polkaDots(const vec2 uv, const float radius = 0.35f) {
    const auto cell = floor(uv + 0.5f);
    if (perlin(cell + 0.5f) > 0.0f) {
        const auto maxShift = 0.5f - radius;
        const auto center = cell + maxShift * vec2{
            perlin(cell + vec2{1.7f, -5.8f}),
            perlin(cell + vec2{-5.5f, 2.6f})
        };
        return distance2(uv, center) < radius * radius;
    }
    return false;
}

CUDAINLINE float calcOctavesAntiAliased(const Vector dpdx, const Vector dpdy) {
    return fmax(0.0f, -1.0f - 0.5f*log2(fmax(length2(dpdx), length2(dpdy))));
}

CUDAINLINE float fbm(const Vector p, const float omega, const float octaves) {
    auto sum = 0.0f, lambda = 1.0f, o = 1.0f;
    const int n = octaves;
    for (auto i = 0; i < n; ++i) {
        sum += o * perlin(lambda * p);
        lambda *= 1.99f;
        o *= omega;
    }
    return sum + o * glm::smoothstep(0.3f, 0.7f, octaves - n) * perlin(lambda * p);
}

CUDAINLINE float turbulence(const Vector p, const float omega, const float octaves,
    const int maxOctaves) {
    auto sum = 0.0f, lambda = 1.0f, o = 1.0f;
    const int n = octaves;
    for (auto i = 0; i < n; ++i) {
        sum += o * fabs(perlin(lambda * p));
        lambda *= 1.99f;
        o *= omega;
    }
    sum += o * glm::mix(glm::smoothstep(0.3f, 0.7f, octaves - n), 0.2f, fabs(perlin(lambda * p)));
    for (auto i = n; i < maxOctaves; ++i) {
        sum += o * 0.2f;
        o *= omega;
    }
    return sum;
}

CUDAINLINE float windyWaves(const Vector p, float octaves) {
    octaves = fmin(6.0f, octaves);
    return fabs(fbm(0.1f * p, 0.5f, 0.5f*octaves)) * fbm(p, 0.5f, octaves);
}

CUDAINLINE RGBSpectrum marble(const Vector p, const float var, const float omega, const float octaves) {
    const auto marble = p.y + var * fbm(p, omega, octaves);
    auto t = 0.5f + 0.5f * sin(marble);
    constexpr float c[][3] = {
        {0.58f, 0.58f, 0.6f},
        {0.58f, 0.58f, .6f},
        {0.58f, 0.58f, 0.6f},
        {0.5f, 0.5f, 0.5f},
        {0.6f, 0.59f, 0.58f},
        {0.58f, 0.58f, 0.6f},
        {0.58f, 0.58f, 0.6f},
        {0.2f, 0.2f, 0.33f},
        {0.58f, 0.58f, 0.6f}
    };
    constexpr float scale = std::extent<decltype(c)>::value;
    const int beg = std::floor(t * scale);
    t = t * scale - beg;
    #define CONSTRUCT(x) Vector(x[0],x[1],x[2])
    return RGBSpectrum{ glm::catmullRom(CONSTRUCT(c[beg]), CONSTRUCT(c[beg + 1]),
        CONSTRUCT(c[beg + 2]), CONSTRUCT(c[beg + 3]), t) };
    #undef CONSTRUCT
}
