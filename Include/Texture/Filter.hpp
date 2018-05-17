#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

using Filter1D = float(*)(float);

//x must be in [-1,1].
DEVICEINLINE float box(const float) {
    return 1.0f;
}

DEVICEINLINE float triangle(const float x) {
    return 1.0f - fabs(x);
}

DEVICEINLINE float gaussian(const float x, const float negAlpha, const float expv) {
    return expf(negAlpha * x * x) - expv;
}

DEVICEINLINE float sinc(const float x) {
    if (x < 1e-5)return 1.0;
    const auto px = pi<float>() * x;
    return sin(px) / px;
}

DEVICEINLINE float windowedSinc(const float x, const float tau = 2) {
    return sinc(x) * sinc(x / tau);
}

using Filter2D = float(*)(vec2);

template <Filter1D Func>
DEVICEINLINE float filter2D(const vec2 p) {
    return Func(p.x) * Func(p.y);
}

DEVICEINLINE float gaussian2D(const vec2 p, const float negAlpha, const float expv) {
    return gaussian(p.x, negAlpha, expv) * gaussian(p.y, negAlpha, expv);
}

DEVICEINLINE float windowedSinc2D(const vec2 p, const float tau = 2) {
    return windowedSinc(p.x, tau)*windowedSinc(p.y, tau);
}
