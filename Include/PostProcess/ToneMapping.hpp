#include <Base/Common.hpp>
#include "Base/Math.hpp"

CUDAINLINE RGB reinhard(RGB color, const float exposure) {
    color = 1.0f-exp(-color*exposure);
    return pow(color, RGB(1.0f/2.2f));
}

//https://zhuanlan.zhihu.com/p/21983679
CUDAINLINE RGB reinhard(RGB color, const float lum, const float grey) {
    color *= grey / lum;
    return color / (1.0f + color);
}

CUDAINLINE RGB CEExp(const RGB color, const float lum) {
    return 1.0f-exp(-lum*color);
}

namespace Impl {
    constexpr auto a = 0.22f, b = 0.30f, c = 0.10f, d = 0.20f, e = 0.01f, f = 0.30f,white = 11.2f;
    CUDAINLINE float func(float x) {
        return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
    }
}

//http://filmicgames.com/archives/75
CUDAINLINE RGB Uncharted(RGB color, float lum) {
    color *= 1.6f*lum;
    return vec3{ Impl::func(color.x),Impl::func(color.y),Impl::func(color.z) } / Impl::func(Impl::white);
}

//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
CUDAINLINE RGB ACES(RGB color, float lum) {
    constexpr auto a = 2.51f,b = 0.03f,c = 2.43f,d = 0.59f,e = 0.14f;
    color *= lum;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
}

CUDAINLINE float calcLum(const float ave, const float alpha = 0.18f) {
    return alpha / exp(ave);
}
