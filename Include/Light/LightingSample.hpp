#pragma once
#include <Core/Common.hpp>
#include <Core/Math.hpp>

template<typename Spectrum>
struct LightingSample final {
    vec3 wi;
    float pdf;
    Spectrum illumination;
    CUDAINLINE LightingSample(const vec3 wi, const Spectrum& illumination, const float pdf = 1.0f)
        :wi(wi), pdf(pdf), illumination(illumination) {}
};
