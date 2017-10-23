#include <Base/Common.hpp>

CUDAInline RGB Reinhard(RGB color,float lum, float grey) {
    color *= grey / lum;
    return color / (1.0f + color);
}

CUDAInline RGB CEExp(RGB color, float lum) {
    return 1.0f-exp(-lum*color);
}

namespace Impl {
    constexpr auto A = 0.22f, B = 0.30f, C = 0.10f, D = 0.20f, E = 0.01f, F = 0.30f;
    constexpr auto f1 = ((1.0f * (A * 1.0f + C * B) + D * E) / (1.0f * (A * 1.0f + B) + D * F)) - E / F;
    CUDAInline RGB func(RGB x) {
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    }
}

CUDAInline RGB Uncharted(RGB color, float lum) {
    const float WHITE = 11.2f;
    return Impl::func(1.6f * lum * color) / Impl::f1;
}

CUDAInline RGB ACES(RGB color, float lum) {
    constexpr float A = 2.51f,B = 0.03f,C = 2.43f,D = 0.59f,E = 0.14f;
    color *= lum;
    return (color * (A * color + B)) / (color * (C * color + D) + E);
}

