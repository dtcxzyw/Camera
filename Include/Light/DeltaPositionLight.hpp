#pragma once
#include <Light/LightingSample.hpp>

class PointLight final {
private:
    Point mPos;
    Spectrum mIllumination, mPower;
public:
    BOTH PointLight() = default;
    BOTH PointLight(const Point pos, const Spectrum& illumination) 
        :mPos(pos), mIllumination(illumination) {
        mPower = mIllumination * (4.0f*pi<float>());
    }
    BOTH LightingSample sampleLi(const vec2,const Point pos) const {
        const auto delta = mPos - pos;
        const auto invDis2 = 1.0f / length2(delta);
        return { delta*sqrt(invDis2),mIllumination*invDis2 };
    }
    BOTH Spectrum power() const {
        return mPower;
    }
};

class SpotLight final {
private:
    Point mPos;
    Transform mWorldToLight;
    Spectrum mIllumination, mPower;
    float mFallOffStart, mWidth, mInvLen;
    BOTH float fallOff(const float cosTheta) const {
        const auto k = clamp((cosTheta - mWidth) *mInvLen, 0.0f, 1.0f);
        const auto k2 = k * k;
        return k2 * k2;
    }
public:
    BOTH SpotLight() = default;
    BOTH SpotLight(const Transform& transform, const Spectrum& illumination,
        const float fallOffStart, const float width) :mPos(transform(Point{})),
        mWorldToLight(inverse(transform)), mIllumination(illumination),
        mFallOffStart(cos(glm::radians(fallOffStart))), mWidth(cos(glm::radians(width))) {
        mPower = mIllumination * (2.0f*pi<float>()*(1.0f - 0.5f*(mFallOffStart + mWidth)));
        mInvLen = 1.0f / (mFallOffStart - mWidth);
    } 
    BOTH LightingSample sampleLi(const vec2, const Point pos) const {
        const auto delta = mPos - pos;
        const auto invDis2 = 1.0f / length2(delta);
        const auto wi= delta * sqrt(invDis2);
        return { wi,mIllumination*invDis2*fallOff(mWorldToLight(wi).z) };
    }
    BOTH Spectrum power() const {
        return mPower;
    }
};
