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

struct Interaction;
struct SurfaceInteraction;

struct LightTag {
    DEVICE Spectrum le(const Ray&) const {
        return Spectrum{};
    }

    DEVICE bool isDelta() const {
        return false;
    }

    void preprocess(const Point&, float) const {}
    DEVICE float pdfLi(const Interaction&, const Vector&) const {
        return 0.0f;
    }

    DEVICE Spectrum emitL(const Interaction&, const Vector&) const {
        return Spectrum{};
    }

    DEVICE bool intersect(const Ray&) const {
        return false;
    }

    DEVICE bool intersect(const Ray&, float&, SurfaceInteraction&) const {
        return false;
    }
};
