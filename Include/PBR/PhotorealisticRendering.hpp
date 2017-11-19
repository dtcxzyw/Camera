#pragma once
#include <cmath>

//https://zhuanlan.zhihu.com/p/23827065
inline float toFOV(float s, float f) {
    return 2.0f*atanf(sqrtf(s)/(2.0f*f));
}
