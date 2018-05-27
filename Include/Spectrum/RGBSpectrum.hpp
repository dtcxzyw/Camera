#pragma once
#include <Core/Common.hpp>
#include <Core/CompileBegin.hpp>
#include <glm/gtx/color_space.hpp>
#include <Core/CompileEnd.hpp>

class RGBSpectrum final {
private:
    RGB mVal;
public:
    BOTH explicit RGBSpectrum(const float v = 0.f) : mVal(v) {}
    BOTH explicit RGBSpectrum(const RGB& v) : mVal(v) {}
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
        return glm::luminosity(mVal);
    }

    BOTH RGB toRGB() const {
        return mVal;
    }

    BOTH friend RGBSpectrum sqrt(const RGBSpectrum& col) {
        return RGBSpectrum{ glm::sqrt(col.mVal) };
    }

    DEVICE RGBSpectrum atomicAdd(const RGBSpectrum& rhs) {
        ::atomicAdd(&mVal.x, rhs.mVal.x);
        ::atomicAdd(&mVal.y, rhs.mVal.y);
        ::atomicAdd(&mVal.z, rhs.mVal.z);
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
