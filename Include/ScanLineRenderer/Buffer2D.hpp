#pragma once
#include  <Base/Math.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>

template<typename T>
class Buffer2DRef final {
private:
    T* mPtr;
    int mOffset;
    CUDAINLINE int toPos(const ivec2 p) const {
        const auto bx=p.x>>5,ox = p.x & 0b11111,by=p.y>>5,oy = p.y & 0b11111;
        return (bx*mOffset + by) << 10 | ox<<5 | oy;
    }
public:
    Buffer2DRef() = default;
    Buffer2DRef(T* buf, const int offset):mPtr(buf), mOffset(offset){}
    CUDAINLINE T& get(const ivec2 uv) {
        return mPtr[toPos(uv)];
    }
    CUDAINLINE T get(const ivec2 uv) const{
        return mPtr[toPos(uv)];
    }
};

template<typename T>
class Buffer2D final:Uncopyable {
private:
    uvec2 mSize;
    DataViewer<T> mData;
public:
    explicit Buffer2D(const uvec2 size) :mSize(size),
        mData(calcBlockSize(size.x,32U)*calcBlockSize(size.y,32U)*1024U) {}
    void clear(CommandBuffer& buffer) {
        buffer.pushOperator([=](Id,ResourceManager&,Stream& stream) {stream.memset(mData, 0xff); });
    }
    auto size() const {
        return mSize;
    }
    Buffer2DRef<T> toBuffer() {
        return { mData.begin(),static_cast<int>(calcBlockSize(mSize.y,32U)) };
    }
};
