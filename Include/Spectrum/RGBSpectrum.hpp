#pragma once
#include <Core/Common.hpp>
#include <Core/Math.hpp>

class RGBSpectrum final:public vec3 {
public:
    BOTH RGBSpectrum(const float v = 0.f) : vec3(v) {}
    BOTH RGBSpectrum(const vec3 &v) : vec3(v) {}
    BOTH float lum() const {
        return luminosity(*this);
    }
    BOTH RGBSpectrum toRGB() const {
        return *this;
    }
};
