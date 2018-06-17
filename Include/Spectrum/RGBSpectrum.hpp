#pragma once
#include <Core/Common.hpp>
#include <Core/IncludeBegin.hpp>
#include <glm/gtx/color_space.hpp>
#include <Core/IncludeEnd.hpp>
#include <Core/DeviceFunctions.hpp>
#include <Spectrum/SpectrumShared.hpp>

class RGBSpectrum final {
private:
    RGB mVal;
public:
    BOTH    explicit RGBSpectrum(const float v = 0.0f) : mVal(v) {}
    BOTH    explicit RGBSpectrum(const RGB& v, SpectrumType = SpectrumType::Reflectance) : mVal(v) {}
    #define OPVEC(op) \
    BOTH RGBSpectrum& operator##op##=(const RGBSpectrum& rhs) {\
        mVal op##= rhs.mVal;\
        return *this;\
    }\
    BOTH RGBSpectrum operator##op(const RGBSpectrum& rhs) const {\
        auto res = *this;\
        return res op##= rhs;\
    }
    OPVEC(+)
    OPVEC(-)
    OPVEC(*)
    OPVEC(/)
    #undef OPVEC

    #define OPFLOAT(op) \
    BOTH RGBSpectrum& operator##op##=(const float rhs) {\
        mVal op##= rhs;\
        return *this;\
    }\
    BOTH RGBSpectrum operator##op(const float rhs) const {\
        auto res = *this;\
        return res op##= rhs;\
    }
    OPFLOAT(+)
    OPFLOAT(-)
    OPFLOAT(*)
    OPFLOAT(/ )
    #undef OPFLOAT

    BOTH    float y() const {
        return dot(mVal, RGB{0.212671f, 0.715160f, 0.072169f});
    }

    DEVICE float maxComp() const {
        return max3(mVal.x, mVal.y, mVal.z);
    }

    BOTH    RGB toRGB() const {
        return mVal;
    }

    BOTH    friend RGBSpectrum sqrt(const RGBSpectrum& col) {
        return RGBSpectrum{glm::sqrt(col.mVal)};
    }

    DEVICE void atomicAdd(const RGBSpectrum& rhs) {
        deviceAtomicAdd(&mVal.x, rhs.mVal.x);
        deviceAtomicAdd(&mVal.y, rhs.mVal.y);
        deviceAtomicAdd(&mVal.z, rhs.mVal.z);
    }
};

#define OPFLOAT(op) \
    inline BOTH RGBSpectrum operator##op(const float lhs,const RGBSpectrum& rhs) {\
        return RGBSpectrum(lhs)##op##rhs;\
    }
OPFLOAT(+)
OPFLOAT(-)
OPFLOAT(*)
OPFLOAT(/ )
#undef OPFLOAT

inline BOTH RGBSpectrum mix(const RGBSpectrum& a, const RGBSpectrum& b, const float w) {
    return a + (b - a) * w;
}
