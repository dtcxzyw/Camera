#pragma once
#include <Base/DispatchSystem.hpp>
#include <device_atomic_functions.h>

template<typename T>
class QueueGPU final {
private:
    T * mAddress;
    unsigned int* mCnt;
public:
    QueueGPU(T* address,unsigned int* cnt):mAddress(address),mCnt(cnt){}
    CUDAINLINE void push(T val) {
        constexpr auto maxv = std::numeric_limits<unsigned int>::max();
        mAddress[atomicInc(mCnt, maxv)] = val;
    }
    CUDAINLINE void push(T* val,unsigned int cnt) {
        const auto base = atomicAdd(mCnt, cnt);
        for (auto i = 0; i < cnt; ++i)
            mAddress[base + i] = val[i];
    }
};

template<typename T>
class Queue final:Uncopyable {
private:
    MemoryRef<T> mData;
    MemoryRef<unsigned int> mCnt;
public:
    Queue(CommandBuffer& buffer, const size_t size)
        :mData(buffer.allocBuffer<T>(size)), mCnt(buffer.allocBuffer<unsigned int>()) {
        buffer.memset(mCnt);
    }
    auto get(CommandBuffer& buffer) const {
        return buffer.makeLazyConstructor<QueueGPU<T>>(mData,mCnt);
    }
    auto size() const {
        return mCnt;
    }
    auto data() const {
        return mData;
    }
    void earlyRelease() {
        mData.earlyRelease();
        mCnt.earlyRelease();
    }
};
