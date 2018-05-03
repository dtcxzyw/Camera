#pragma once
#include <Math/Geometry.hpp>
#include <Spectrum/SpectrumConfig.hpp>

struct LightingSample final {
    Normal wi;
    float pdf;
    Spectrum illumination;
    CUDA LightingSample() = default;
    CUDA LightingSample(const Vector& wi, const Spectrum& illumination, const float pdf = 1.0f)
        :wi(makeNormalUnsafe(wi)), pdf(pdf), illumination(illumination) {}
};
