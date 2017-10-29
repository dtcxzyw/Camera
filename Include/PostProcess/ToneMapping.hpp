#include <Base/Common.hpp>

CUDAInline RGB Reinhard(RGB color, float exposure) {
    color = 1.0f-exp(-color*exposure);
    return pow(color, RGB(1.0f/2.2f));
}

//https://zhuanlan.zhihu.com/p/21983679
CUDAInline RGB Reinhard(RGB color,float lum, float grey) {
    color *= grey / lum;
    return color / (1.0f + color);
}

CUDAInline RGB CEExp(RGB color, float lum) {
    return 1.0f-exp(-lum*color);
}

namespace Impl {
    constexpr auto A = 0.22f, B = 0.30f, C = 0.10f, D = 0.20f, E = 0.01f, F = 0.30f,white = 11.2f;
    CUDAInline float func(float x) {
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    }
}

//http://filmicgames.com/archives/75
CUDAInline RGB Uncharted(RGB color, float lum) {
    color *= 1.6f*lum;
    return vec3{ Impl::func(color.x),Impl::func(color.y),Impl::func(color.z) } / Impl::func(Impl::white);
}

//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
CUDAInline RGB ACES(RGB color, float lum) {
    constexpr float A = 2.51f,B = 0.03f,C = 2.43f,D = 0.59f,E = 0.14f;
    color *= lum;
    return clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0f, 1.0f);
}

inline float calcLum(float ave, float alpha = 0.18f) {
    return alpha / exp(ave);
}
