#pragma once
#include <Light/LightingSample.hpp>

class DistantLight final {
private:
    Vector mDirection, mOffset;
    Spectrum mIllumination;
public:
    DistantLight() = default;
    BOTH DistantLight(const Transform& trans, const Vector& dir, const Spectrum& illumination,
        const float worldRadius)
        : mDirection(normalize(inverse(trans)(dir))), mOffset(2.0f * worldRadius * mDirection),
        mIllumination(illumination) {}

    DEVICE LightingSample sampleLi(const vec2, const Point& pos) const {
        return LightingSample{mDirection, mIllumination, pos - mOffset};
    }

    DEVICE Spectrum le(const Ray&) const {
        return Spectrum{};
    }

    DEVICE bool isDelta() const {
        return true;
    }
};
