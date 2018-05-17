#pragma once
#include <Math/Math.hpp>

DEVICEINLINE vec2 concentricSampleDisk(vec2 p) {
    p = 2.0f*p - 1.0f;
    if (p.x == 0.0f & p.y == 0.0f)return { 0.0f,0.0f };
    float r, theta;
    if (fabs(p.x) > fabs(p.y)) {
        r = p.x;
        theta = quarter_pi<float>()*p.y / p.x;
    }
    else {
        r = p.y;
        theta = half_pi<float>() - quarter_pi<float>()*p.x / p.y;
    }
    return r * vec2{ cos(theta),sin(theta) };
}

DEVICEINLINE Vector cosineSampleHemisphere(const vec2 p) {
    const auto d = concentricSampleDisk(p);
    return { d.x,d.y ,sqrt(1.0f - d.x*d.x - d.y*d.y) };
}
