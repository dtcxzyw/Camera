#pragma once
#include <Base/Pipeline.hpp>
#include <stdexcept>

namespace Impl {
    constexpr auto blockNum = 512U, blockSize = 32U;
    void* constantAlloc(unsigned int size);
    void constantFree(void* address, unsigned int size);
    void constantSet(void* dest, const void* src, unsigned int size, cudaStream_t stream);
}

template <typename T>
class Constant final : Uncopyable {
private:
    T* mAddress;
public:
    Constant(): mAddress(static_cast<T*>(Impl::constantAlloc(sizeof(T)))) {
        if (!mAddress)
            throw std::runtime_error("Failed to allocate CUDA Constant Memory.");
    }

    explicit Constant(const T& rhs) : Constant() {
        set(rhs);
    }

    Constant(Constant&& rhs) noexcept: mAddress(rhs.mAddress) {
        rhs.mAddress = nullptr;
    }

    Constant& operator=(Constant&& rhs) noexcept {
        if (this != &rhs) {
            Impl::constantFree(mAddress, sizeof(T));
            mAddress = rhs.mAddress;
            rhs.mAddress = nullptr;
        }
        return *this;
    }

    T* get() const {
        return mAddress;
    }

    void set(const T& rhs, Stream& stream) {
        Impl::constantSet(mAddress, &rhs, sizeof(T), stream.get());
    }

    ~Constant() {
        if (mAddress)Impl::constantFree(mAddress, sizeof(T));
    }
};
