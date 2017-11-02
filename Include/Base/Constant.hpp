#pragma once
#include "Pipeline.hpp"

namespace Impl {
    constexpr auto blockNum = 512U, blockSize = 32U;
    void* constantAlloc(unsigned int size);
    void constantFree(void* address,unsigned int size);
    void constantSet(void* dest,const void* src,unsigned int size,cudaStream_t pipeline);
}

template<typename T>
class Constant final:Uncopyable {
private:
    T* const mAddress;
public:
    Constant():mAddress(static_cast<T*>(Impl::constantAlloc(sizeof(T)))) {
        if (!mAddress)
            throw std::exception("Failed to allocate CUDA Constant Memory.");
    }

    Constant(const T& rhs) :Constant() {
        set(rhs);
    }

    T* get() const {
        return mAddress;
    }

    void set(const T& rhs,Pipeline& pipeline) {
        Impl::constantSet(mAddress, &rhs,sizeof(T),pipeline.getId());
    }

    ~Constant() {
        Impl::constantFree(mAddress,sizeof(T));
    }
};
