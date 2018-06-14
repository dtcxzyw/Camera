#pragma once
#include <Core/CommandBuffer.hpp>
#include <Core/DeviceFunctions.hpp>

template <typename T>
class QueueRef final {
private:
    T* mAddress;
    uint32_t* mCnt;
public:
    QueueRef(T* address, uint32_t* cnt): mAddress(address), mCnt(cnt) {}
    DEVICEINLINE void push(T val) {
        constexpr auto maxv = std::numeric_limits<uint32_t>::max();
        mAddress[deviceAtomicInc(mCnt, maxv)] = val;
    }

    DEVICEINLINE void push(T* val, uint32_t cnt) {
        const auto base = deviceAtomicAdd(mCnt, cnt);
        for (auto i = 0; i < cnt; ++i)
            mAddress[base + i] = val[i];
    }
};

template <typename T>
class Queue final : Uncopyable {
private:
    Span<T> mData;
    Span<uint32_t> mCnt;
public:
    Queue(CommandBuffer& buffer, const size_t size)
        : mData(buffer.allocBuffer<T>(size)), mCnt(buffer.allocBuffer<uint32_t>()) {
        buffer.memset(mCnt);
    }

    auto get(CommandBuffer& buffer) const {
        return buffer.makeLazyConstructor<QueueRef<T>>(mData, mCnt);
    }

    auto size() const {
        return mCnt;
    }

    auto data() const {
        return mData;
    }
};
