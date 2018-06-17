#pragma once
#include <Light/Light.hpp>
#include <Math/Interaction.hpp>

class PointLight final : public LightTag {
private:
    Point mPos;
    Spectrum mIllumination;
public:
    PointLight() = default;

    PointLight(const Point pos, const Spectrum& illumination)
        : mPos(pos), mIllumination(illumination) {}

    DEVICE LightingSample sampleLi(const vec2, const Interaction& interaction) const {
        const auto delta = mPos - interaction.pos;
        const auto invDis2 = 1.0f / length2(delta);
        return {delta * sqrt(invDis2), mIllumination * invDis2, mPos};
    }

    DEVICE bool isDelta() const {
        return true;
    }
};

class SpotLight final : public LightTag {
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

    SpotLight(const Transform& transform, const Spectrum& illumination,
        const float fallOffStart, const float width) : mPos(transform(Point{})),
        mWorldToLight(inverse(transform)), mIllumination(illumination),
        mFallOffStart(cos(glm::radians(fallOffStart))), mWidth(cos(glm::radians(width))) {
        mInvLen = 1.0f / (mFallOffStart - mWidth);
    }

    DEVICE LightingSample sampleLi(const vec2, const Interaction& interaction) const {
        const auto delta = mPos - interaction.pos;
        const auto invDis2 = 1.0f / length2(delta);
        const auto wi = delta * sqrt(invDis2);
        return {wi, mIllumination * invDis2 * fallOff(mWorldToLight(wi).z), mPos};
    }

    DEVICE bool isDelta() const {
        return true;
    }
};
