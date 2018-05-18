#pragma once
#include <Core/CommandBuffer.hpp>

namespace Impl {
    template<typename T>
    GLOBAL void updateCache(const unsigned int size, T* address) {
        const auto id = getId();
        if (id >= size)return;
        address[id] <<= 1;
    }
}

template<typename T>
class RenderingCacheBlockRef final {
private:
    T * mAddress;
    T* mBegin;
    T* mEnd;
public:
    RenderingCacheBlockRef(T* address, T* begin, T* end)
        :mAddress(address), mBegin(begin), mEnd(end) {}
    DEVICEINLINE RenderingCacheBlockRef(){}
    DEVICEINLINE bool query(const unsigned int id) const {
        auto ptr = mAddress + id;
        return (mBegin <= ptr & ptr < mEnd) | (*ptr);
    }
    DEVICEINLINE void record(const unsigned int id) const {
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
    RenderingCacheBlock(T* address, T* begin, T* end, const bool isBlock=false)
        :mAddress(address), mBegin(begin), mEnd(end),mIsBlock(isBlock) {}
    void update(CommandBuffer& buffer) {
        buffer.launchKernelLinear(Impl::updateCache<T>,mEnd-mBegin,mBegin);
    }
    auto toBlock() const {
        return RenderingCacheBlockRef<T>{mAddress,mBegin,mEnd};
    }
    auto isBlock() const {
        return mIsBlock;
    }
};

template<typename T>
class RenderingCache final:Uncopyable {
private:
    MemorySpan<T> mData;
    std::queue<RenderingCacheBlock<T>> mBlocks;
    unsigned int mBlockSize;
    bool mShouldReset;
public:
    using Block = RenderingCacheBlock<T>;
    using BlockRef = RenderingCacheBlockRef<T>;
    void reset() {
        mShouldReset = true;
    }

    explicit RenderingCache(const size_t size, const size_t blockNum = 30)
        :mData(MemorySpan<T>(size)),mBlockSize(std::max(static_cast<size_t>(1), size / blockNum)) {
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
