#pragma once
#include <Base/Common.hpp>
#include <Base/CompileBegin.hpp>
#include <device_functions.h>
#include <Base/CompileEnd.hpp>

CUDAINLINE float dist(const float dis) {
    return 1.0f / (dis*dis);
}

CUDAINLINE float distUE4(const float dis2, const float r2) {
    const auto ratio = dis2 / r2;
    const auto k = saturate(1.0f - ratio * ratio);
    return k*k /(dis2+1.0f);
}

CUDAINLINE float distPIXAR(const float r, const float m, const float k, const float l, const float alpha) {
    const auto s = log(k / m);
    const auto beta = -alpha / s;
    return r < l ? m * exp(s*pow(r / l, beta)) : k * pow(l / r, alpha);
}

