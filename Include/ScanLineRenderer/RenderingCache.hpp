#pragma once
#include <Base/DispatchSystem.hpp>

namespace Impl {
    template<typename T>
    CALLABLE void updateCache(unsigned int size, T* address) {
        auto id = getID();
        if (id >= size)return;
        address[id] <<= 1;
    }
}

template<typename T>
class RenderingCacheBlockGPU final {
private:
    T * mAddress;
    T* mBegin;
    T* mEnd;
public:
    RenderingCacheBlockGPU(T* address, T* begin, T* end)
        :mAddress(address), mBegin(begin), mEnd(end) {}
    CUDAInline RenderingCacheBlockGPU(){}
    CUDAInline bool query(unsigned int id) {
        auto ptr = mAddress + id;
        return (mBegin <= ptr & ptr < mEnd) | (*ptr);
    }
    CUDAInline void record(unsigned int id) {
        mAddress[id] =std::numeric_limits<T>::max();
    }
};

template<typename T>
class RenderingCacheBlock final {
private:
    T* mAddress;
    T* mBegin;
    T* mEnd;
    bool mIsBlock;
public:
    RenderingCacheBlock(T* address, T* begin, T* end,bool isBlock=false)
        :mAddress(address), mBegin(begin), mEnd(end),mIsBlock(isBlock) {}
    void update(CommandBuffer& buffer) {
        buffer.runKernelLinear(Impl::updateCache<T>,mEnd-mBegin,mBegin);
    }
    auto toBlock() const {
        return RenderingCacheBlockGPU<T>{mAddress,mBegin,mEnd};
    }
    auto isBlock() const {
        return mIsBlock;
    }
};

template<typename T>
class RenderingCache final:Uncopyable {
private:
    DataViewer<T> mData;
    std::queue<RenderingCacheBlock<T>> mBlocks;
    unsigned int mBlockSize;
    bool mShouldReset;
public:
    using Block = RenderingCacheBlock<T>;
    using BlockGPU = RenderingCacheBlockGPU<T>;
    void reset() {
        mShouldReset = true;
    }
    RenderingCache(size_t size, size_t blockNum = 30)
        :mData(allocBuffer<T>(size)),mBlockSize(std::max(static_cast<size_t>(1), size / blockNum)) {
        auto begin = mData.begin();
        auto end = begin + mBlockSize;
        while (end < mData.end()) {
            mBlocks.emplace(mData.begin(),begin, end,true);
            begin += mBlockSize;
            end += mBlockSize;
        }
        mBlocks.emplace(mData.begin(),begin, mData.end(),true);
        reset();
    }
    auto blockSize() const {
        return mBlockSize;
    }
    auto pop(CommandBuffer& buffer) {
        if (mShouldReset) {
            mShouldReset = false;
            return Block{mData.begin(),mData.begin(),mData.end()};
        }
        if (mBlocks.empty())return Block{ mData.begin(),nullptr,nullptr };
        Block block = mBlocks.front();
        block.update(buffer);
        mBlocks.pop();
        return block;
    }
    void push(Block block) {
        if(block.isBlock())
            mBlocks.push(block);
    }
};

using RC8 = RenderingCache<uint8>;
using RC16 = RenderingCache<uint16>;
using RC32 = RenderingCache<uint32>;
using RC64 = RenderingCache<uint64>;
