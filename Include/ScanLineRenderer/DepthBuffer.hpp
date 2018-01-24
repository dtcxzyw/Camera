#pragma once
#include <Base/Pipeline.hpp>
#include <Base/DispatchSystem.hpp>
#include <device_atomic_functions.h>

template<typename T>
class DepthBufferGPU final {
private:
    T* mPtr;
    int mOffset;
    CUDAInline int toPos(ivec2 p) const {
        return p.y*mOffset + p.x;
    }
public:
    DepthBufferGPU() = default;
    DepthBufferGPU(T* buf, int offset):mPtr(buf), mOffset(offset){}
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
        return { mData.begin(),static_cast<int>(mSize.x) };
    }
};
