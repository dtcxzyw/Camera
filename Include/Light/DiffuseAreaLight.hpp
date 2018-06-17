#pragma once
#include <Light/Light.hpp>
#include <Light/Shapes/ShapeWrapper.hpp>

class DiffuseAreaLight final : public LightTag {
private:
    Spectrum mEmit;
    ShapeWrapper mShape;
    bool mTwoSided;
public:
    DiffuseAreaLight(const Spectrum& emit, const ShapeWrapper& shape, const bool twoSided = false)
        : mEmit(emit), mShape(shape), mTwoSided(twoSided) {}

    DEVICE Spectrum emitL(const Interaction& isect, const Vector& w) const {
        return mTwoSided | dot(Vector{isect.localGeometry.normal}, w) > 0.0f ? mEmit : Spectrum{};
    }

    DEVICE LightingSample sampleLi(const vec2 sample, const Interaction& isect) const {
        float pdf;
        const auto pSpape = mShape.sample(isect, sample, pdf);
        if (pdf == 0.0f | distance2(pSpape.pos, isect.pos) == 0.0f) {
            pdf = 0.0f;
            return LightingSample{};
        }
        const auto wi = normalize(pSpape.pos - isect.pos);
        return LightingSample{wi, emitL(pSpape, -wi), pSpape.pos, pdf};
    }

    DEVICE float pdfLi(const Interaction& isect, const Vector& wi) const {
        return mShape.pdf(isect, wi);
    }

    DEVICE bool intersect(const Ray& ray) const {
        return mShape.intersect(ray);
    }

    DEVICE bool intersect(const Ray& ray, float& tHit, SurfaceInteraction& isect) const {
        return mShape.intersect(ray,tHit,isect);
    }
};
