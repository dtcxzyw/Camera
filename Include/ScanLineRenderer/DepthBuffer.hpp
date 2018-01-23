#pragma once
#include <Base/Pipeline.hpp>
#include <Base/DispatchSystem.hpp>
#include <device_atomic_functions.h>

template<typename T>
class DepthBufferGPU final {
private:
    T* mPtr;
    ivec3 mInfo;
    CUDAInline int toPos(ivec2 p) const {
        p.x = clamp(p.x, 0, mInfo.x);
        p.y = clamp(p.y, 0, mInfo.y);
        return p.y*mInfo.z + p.x;
    }
public:
    DepthBufferGPU() = default;
    DepthBufferGPU(T* buf, ivec3 info):mPtr(buf), mInfo(info){}
    CUDAInline T get(ivec2 uv) const {
        return mPtr[toPos(uv)];
    }
    CUDAInline void set(ivec2 uv, T z) const {
        atomicMin(mPtr+toPos(uv),z);
    }
};

template<typename T>
class DepthBuffer final:Uncopyable {
private:
    DataViewer<T> mData;
    uvec2 mSize;
public:
    DepthBuffer(uvec2 size) :mSize(size),mData(allocBuffer<T>(size.x*size.y)) {}
    void clear(CommandBuffer& buffer) {
        buffer.pushOperator([=](Stream& stream) {stream.memset(mData, 0xff); });
    }
    DepthBufferGPU<T> toBuffer() {
        return { mData.begin(),{mSize.x-1,mSize.y-1,mSize.x} };
    }
};
