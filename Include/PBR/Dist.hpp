#pragma once
#include <Base/Common.hpp>
#include <device_functions.h>

CUDAInline float dist(float dis) {
    return 1.0f / (dis*dis);
}

CUDAInline float distUE4(float dis2,float r2) {
    auto ratio = dis2 / r2;
    auto k = saturate(1.0f - ratio * ratio);
    return k*k /(dis2+1.0f);
}

CUDAInline float distPIXAR(float r, float m, float k, float l, float alpha) {
    float s = log(k / m);
    float beta = -alpha / s;
    return r < l ? m * exp(s*pow(r / l, beta)) : k * pow(l / r, alpha);
}

