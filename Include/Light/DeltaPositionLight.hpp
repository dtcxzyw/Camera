#pragma once
#include <Light/LightingSample.hpp>

class PointLight final {
private:
    Point mPos;
    Spectrum mIllumination;
public:
    PointLight() = default;
    BOTH PointLight(const Point pos, const Spectrum& illumination)
        : mPos(pos), mIllumination(illumination) {}

    DEVICE LightingSample sampleLi(const vec2, const Point pos) const {
        const auto delta = mPos - pos;
        const auto invDis2 = 1.0f / length2(delta);
        return {delta * sqrt(invDis2), mIllumination * invDis2, mPos};
    }

    DEVICE Spectrum le(const Ray&) const {
        return Spectrum{};
    }

    DEVICE bool isDelta() const {
        return true;
    }
};

class SpotLight final {
private:
    Point mPos;
    Transform mWorldToLight;
    Spectrum mIllumination;
    float mFallOffStart, mWidth, mInvLen;
    DEVICE float fallOff(const float cosTheta) const {
        const auto k = clamp((cosTheta - mWidth) * mInvLen, 0.0f, 1.0f);
        const auto k2 = k * k;
        return k2 * k2;
    }

public:
    SpotLight() = default;
    BOTH SpotLight(const Transform& transform, const Spectrum& illumination,
        const float fallOffStart, const float width) : mPos(transform(Point{})),
        mWorldToLight(inverse(transform)), mIllumination(illumination),
        mFallOffStart(cos(glm::radians(fallOffStart))), mWidth(cos(glm::radians(width))) {
        mInvLen = 1.0f / (mFallOffStart - mWidth);
    }

    DEVICE LightingSample sampleLi(const vec2, const Point pos) const {
        const auto delta = mPos - pos;
        const auto invDis2 = 1.0f / length2(delta);
        const auto wi = delta * sqrt(invDis2);
        return {wi, mIllumination * invDis2 * fallOff(mWorldToLight(wi).z), mPos};
    }

    DEVICE Spectrum le(const Ray&) const {
        return Spectrum{};
    }

    DEVICE bool isDelta() const {
        return true;
    }
};
