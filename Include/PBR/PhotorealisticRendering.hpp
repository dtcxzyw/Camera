#pragma once
#include <cmath>
//https://zhuanlan.zhihu.com/p/23827065

inline float toFOV(float s, float f) {
    return 2.0f*atanf(sqrtf(s)/(2.0f*f));
}

CUDAInline float calcZ(float z, float znear, float zfar) {
    return zfar * znear / (zfar-z * (zfar - znear));
}

//https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch23.html
CUDAInline float DoFRadius(float z, float znear, float zfar,float f,float aperture,float df) {
    auto CoCScale = (aperture * f * df * (zfar - znear)) /((df - f) * znear * zfar);
    auto CoCBias = (aperture * f * (znear - df)) /((df * f) * znear);
    auto CoC = fabs(z * CoCScale + CoCBias);
    return CoC;
}
