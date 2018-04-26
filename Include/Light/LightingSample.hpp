#pragma once
#include <Core/Common.hpp>
#include <Math/Geometry.hpp>

template<typename Spectrum>
struct LightingSample final {
    Normal wi;
    float pdf;
    Spectrum illumination;
    CUDAINLINE LightingSample(const Vector wi, const Spectrum& illumination, const float pdf = 1.0f)
        :wi(makeNormalUnsafe(wi)), pdf(pdf), illumination(illumination) {}
};
