#pragma once
#include <Light/LightingSample.hpp>

template<typename Spectrum>
class PointLight final {
private:
    vec3 mPos;
    Spectrum mIllumination, mPower;
public:
    BOTH PointLight() = default;
    BOTH PointLight(const vec3 pos, const Spectrum& illumination) 
        :mPos(pos), mIllumination(illumination) {
        mPower = mIllumination * (4.0f*pi<float>());
    }
    BOTH LightingSample<Spectrum> sample(const vec2,const vec3 pos) const {
        const auto delta = mPos - pos;
        const auto invDis2 = 1.0f / length2(delta);
        return { delta*sqrt(invDis2),mIllumination*invDis2 };
    }
    BOTH Spectrum power() const {
        return mPower;
    }
};

template<typename Spectrum>
class SpotLight final {
private:
    vec3 mPos;
    mat3 mWorldToLight;
    Spectrum mIllumination, mPower;
    float mFallOffStart, mWidth, mInvLen;
    BOTH float fallOff(const float cosTheta) const {
        const auto k = clamp((cosTheta - mWidth) *mInvLen, 0.0f, 1.0f);
        const auto k2 = k * k;
        return k2 * k2;
    }
public:
    BOTH SpotLight() = default;
    BOTH SpotLight(const vec3 pos, const mat4& transform, const Spectrum& illumination,
        const float fallOffStart, const float width)
        :mPos(pos), mWorldToLight(inverse(mat3(transpose(inverse(transform))))),
        mIllumination(illumination),
        mFallOffStart(cos(radians(fallOffStart))), mWidth(cos(radians(width))) {
        mPower = mIllumination * (2.0f*pi<float>()*(1.0f - 0.5f*(mFallOffStart + mWidth)));
        mInvLen = 1.0f / (mFallOffStart - mWidth);
    } 
    BOTH LightingSample<Spectrum> sample(const vec2, const vec3 pos) const {
        const auto delta = mPos - pos;
        const auto invDis2 = 1.0f / length2(delta);
        const auto wi= delta * sqrt(invDis2);
        return { wi,mIllumination*invDis2*fallOff((mWorldToLight*wi).z) };
    }
    BOTH Spectrum power() const {
        return mPower;
    }
};


