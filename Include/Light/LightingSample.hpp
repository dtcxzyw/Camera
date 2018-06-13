#pragma once
#include <Math/Geometry.hpp>
#include <Spectrum/SpectrumConfig.hpp>

struct LightingSample final {
    Point src;
    Vector wi;
    float pdf;
    Spectrum illumination;
    LightingSample() = default;
    DEVICE LightingSample(const Vector& wi, const Spectrum& illumination, const Point& src,
        const float pdf = 1.0f) : src(src), wi(wi), pdf(pdf), illumination(illumination) {}
};
