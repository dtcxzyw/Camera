#pragma once
#include "Common.hpp"

template<typename T>
class Sampler final {
private:
    uvec2 mSize,mClamp;
    const T* ReadOnly mPtr;
    CUDA float weight(float dx, float dy) {
        return abs(dx)*abs(dy);
    }
public:
    Sampler() = default;
    Sampler(T* ptr, uvec2 size) :mPtr(ptr), mSize(size) { 
        mClamp = { size.x - 1U,size.y - 1U };
    }
    BOTH uvec2 size() const {
        return mSize;
    }
    CUDA T get(vec2 p) {
        vec2 rp{ p.x*mSize.x,p.y*mSize.y };
        uint px[2], py[2];
        px[0] = rp.x + 0.5f, px[1] = rp.x - 0.5f;
        py[0] = rp.y + 0.5f, py[1] = rp.y - 0.5f;
        for (int i = 0; i < 2; ++i) {
            px[i] = clamp(px[i], 0U, mClamp.x);
            py[i] = clamp(py[i], 0U, mClamp.y);
        }
        T res = {};
        float sum = 0.0f;
        for(int i=0;i<2;++i)
            for (int j = 0; j < 2; ++j) {
                float w = weight(px[i] - rp.x, py[j] - rp.y);
                sum += w;
                res += mPtr[py[j] * mSize.x + px[i]] * w;
            }
        return res/sum;
    }
    CUDA T getSingle(vec2 p) {
        unsigned int rpx = p.x*mSize.x, rpy = p.y*mSize.y;
        rpx = clamp(rpx, 0U, mClamp.x);
        rpy = clamp(rpy, 0U, mClamp.y);
        return mPtr[rpy*mSize.x + rpx];
    }
};
