#pragma once
#include <Base/Common.hpp>
#include <Base/CompileBegin.hpp>
#include <device_functions.h>
#include <Base/CompileEnd.hpp>

CUDAINLINE float dist(float dis) {
    return 1.0f / (dis*dis);
}

CUDAINLINE float distUE4(float dis2,float r2) {
    auto ratio = dis2 / r2;
    auto k = saturate(1.0f - ratio * ratio);
    return k*k /(dis2+1.0f);
}

CUDAINLINE float distPIXAR(float r, float m, float k, float l, float alpha) {
    float s = log(k / m);
    float beta = -alpha / s;
    return r < l ? m * exp(s*pow(r / l, beta)) : k * pow(l / r, alpha);
}

