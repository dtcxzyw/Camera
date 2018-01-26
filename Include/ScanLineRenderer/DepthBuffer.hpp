#pragma once
#include  <Base/Math.hpp>
#include <Base/Pipeline.hpp>
#include <Base/DispatchSystem.hpp>
#include <device_atomic_functions.h>

template<typename T>
class DepthBufferGPU final {
private:
    T* mPtr;
    int mOffset;
    CUDAINLINE int toPos(ivec2 p) const {
        const auto bx=p.x>>5,ox = p.x & 0b11111,by=p.y>>5,oy = p.y & 0b11111;
        return ((bx*mOffset + by) << 10) | (ox<<5) | oy;
    }
public:
    DepthBufferGPU() = default;
    DepthBufferGPU(T* buf, const int offset):mPtr(buf), mOffset(offset){}
    CUDAINLINE T get(const ivec2 uv) const {
        return mPtr[toPos(uv)];
    }
    CUDAINLINE void set(const ivec2 uv, T z) const {
        atomicMin(mPtr+toPos(uv),z);
    }
};

template<typename T>
class DepthBuffer final:Uncopyable {
private:
    uvec2 mSize;
    DataViewer<T> mData;
public:
    explicit DepthBuffer(const uvec2 size) :mSize(size),
        mData(allocBuffer<T>(calcSize(size.x,32U)*calcSize(size.y,32U)*1024U)) {}
    void clear(CommandBuffer& buffer) {
        buffer.pushOperator([=](Stream& stream) {stream.memset(mData, 0xff); });
    }
    auto size() const {
        return mSize;
    }
    DepthBufferGPU<T> toBuffer() {
        return { mData.begin(),static_cast<int>(calcSize(mSize.y,32U)) };
    }
};
