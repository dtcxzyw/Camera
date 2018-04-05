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
    MemoryRef<T> mData;
    CommandBuffer& mBuffer;
public:
    explicit Buffer2D(CommandBuffer& buffer,const uvec2 size) :mSize(size),
        mData(buffer.allocBuffer<T>(calcBlockSize(size.x, 32U)*calcBlockSize(size.y, 32U) * 1024U)),
        mBuffer(buffer) {}
    void clear(int mask) {
        mBuffer.memset(mData, mask);
    }
    auto size() const {
        return mSize;
    }
    auto toBuffer() const {
        return mBuffer.makeLazyConstructor<Buffer2DRef<T>>
            (mData, static_cast<int>(calcBlockSize(mSize.y, 32U)));
    }
};