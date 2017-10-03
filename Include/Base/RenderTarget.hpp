#pragma once
#include "Pipeline.hpp"

namespace Impl{
    template<typename T>
    CALLABLE void doClear(unsigned int size,T* ptr,T val) {
        ptr[min(getID(),size-1)] = val;
    }
}

template<typename T>
class RenderTarget;

template<typename T>
class RenderRECT final {
private:
    T* mPoints[4];
    vec4 mWeights;
    friend class RenderTarget<T>;
public:
    CUDA void operator=(T v) {
        for (int i = 0; i < 4; ++i)
            mix(*(mPoints[i]), v*(mWeights[i]));
    }
    CUDA T get() const {
        T res = {};
        for (int i = 0; i < 4; ++i)
            res+=(*(mPoints[i]))*(mWeights[i]);
        return res;
    }
    CUDA void clear() {
        *this = -get();
    }
};

template<typename T>
class SingleSample final {
private:
    T* mPoint;
    friend class RenderTarget<T>;
public:
    CUDA void operator=(T v) {
        mix(*mPoint, v);
    }
    CUDA T& get() const {
        return *mPoint;
    }
    CUDA void clear() {
        *this = -get();
    }
};

template<typename T>
class RenderTarget final{
private:
    uvec2 mSize,mClamp;
    T* mPtr;
    CUDA float weight(float dx, float dy) {
        return abs(dx)*abs(dy);
    }
public:
    RenderTarget() = default;
    RenderTarget(T* ptr, uvec2 size) :mPtr(ptr), mSize(size){
        mClamp = { size.x - 1U,size.y - 1U };
    }
    HOST void clear(Pipeline& pipeline,T val = {}) {
        pipeline.run(Impl::doClear<T>,mSize.x*mSize.y, mPtr,val);
    }
    BOTH uvec2 size() const {
        return mSize;
    }
    CUDA RenderRECT<T> get(vec2 p) {
        uint px[2], py[2];
        px[0] = p.x + 0.5f,px[1] = p.x - 0.5f;
        py[0] = p.y + 0.5f,py[1] = p.y - 0.5f;
        for (int i = 0; i < 2; ++i) {
            px[i] = clamp(px[i], 0U,mClamp.x);
            py[i] = clamp(py[i], 0U, mClamp.y);
        }
        RenderRECT<T> res;
        for (int i = 0; i < 2; ++i) 
            for (int j = 0; j < 2; ++j) {
                int id = i << 1 | j;
                res.mPoints[id] = &mPtr[py[j] * mSize.x + px[i]];
                res.mWeights[id] = weight(px[i] - p.x, py[j] - p.y);
            }
        res.mWeights /= res.mWeights.x + res.mWeights.y + res.mWeights.z + res.mWeights.w;
        return res;
    }
    CUDA SingleSample<T> getSingle(vec2 p) {
        unsigned int rpx=p.x,rpy=p.y;
        rpx = clamp(rpx, 0U, mClamp.x);
        rpy = clamp(rpy, 0U, mClamp.y);
        SingleSample<T> res;
        res.mPoint = &mPtr[rpy*mSize.x + rpx];
        return res;
    }
};
