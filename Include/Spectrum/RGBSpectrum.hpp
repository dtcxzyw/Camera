#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

class RGBSpectrum final {
private:
    vec3 mVal;
public:
    BOTH explicit RGBSpectrum(const float v = 0.f) : mVal(v) {}
    BOTH explicit RGBSpectrum(const vec3& v) : mVal(v) {}
    #define OPVEC(op) \
    BOTH RGBSpectrum& operator##op##=(const RGBSpectrum& rhs) {\
        mVal op##= rhs.mVal;\
        return *this;\
    }\
    BOTH RGBSpectrum operator##op(const RGBSpectrum& rhs) const {\
        auto res = *this;\
        return res op##= rhs;\
    }\

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
    }\

        OPFLOAT(+)
        OPFLOAT(-)
        OPFLOAT(*)
        OPFLOAT(/ )
#undef OPFLOAT

    BOTH float lum() const {
        return luminosity(mVal);
    }

    BOTH vec3 toRGB() const {
        return mVal;
    }
};

#define OPFLOAT(op) \
    BOTH RGBSpectrum operator##op(const float lhs,const RGBSpectrum& rhs) {\
        return RGBSpectrum(lhs)##op##rhs;\
    }\

OPFLOAT(+)
OPFLOAT(-)
OPFLOAT(*)
OPFLOAT(/ )
#undef OPFLOAT

BOTH RGBSpectrum mix(const RGBSpectrum& a,const RGBSpectrum& b,const float w) {
    return a + (b - a)*w;
}
