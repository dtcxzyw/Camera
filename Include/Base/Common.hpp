#pragma once
#include <memory>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <tuple>
#define CUDA __device__
#define CUDAInline inline CUDA
#define HOST __host__
#define BOTH CUDA HOST
#define CALLABLE __global__
#define ALIGN __align__(16)
#define ReadOnly __restrict__
#include "Types.hpp"

struct Uncopyable {
    Uncopyable() = default;
    Uncopyable(const Uncopyable&) = delete;
    Uncopyable(Uncopyable&&) = default;
    Uncopyable& operator=(const Uncopyable&) = delete;
    Uncopyable& operator=(Uncopyable&&) = default;
};

class Singletion:Uncopyable {
protected:
    Singletion()=default;
public:
    Singletion(Singletion&&) = delete;
    Singletion& operator=(Singletion&&) = delete;
};

void checkError(cudaError_t error);
void checkError();

class Memory final:Uncopyable{
public:
    Memory(size_t size);
    ~Memory();
    char* getPtr() const;
    size_t size() const;
private:
    void* mPtr;
    size_t mSize;
};

using UniqueMemory = std::unique_ptr<Memory>;
using SharedMemory = std::shared_ptr<Memory>;

template<typename T>
class DataViewer final {
private:
    SharedMemory mMem;
    T* mPtr;
    size_t mSize;
public:
    DataViewer() = default;
    DataViewer(SharedMemory memory, size_t offset = 0, size_t size = 0)
        :mMem(memory),mPtr(reinterpret_cast<T*>(memory->getPtr() + offset)),mSize(size) {
        if (size == 0)
            mSize = (mMem->size()-offset)/sizeof(T);
    }
    template<typename U>
    DataViewer<U> viewAs() const {
        auto rsize = mSize * sizeof(T);
        if (rsize%sizeof(U))
            throw std::invalid_argument("T can not cast to U.");
        return DataViewer<U>(mMem,
            reinterpret_cast<char*>(mPtr) - mMem->getPtr(), rsize / sizeof(U));
    }
    T* begin() const {
        return mPtr;
    }
    T* end() const {
        return mPtr + mSize;
    }
    decltype(auto) operator[](size_t i) {
        return *(mPtr + i);
    }
    decltype(auto) operator[](size_t i) const {
        return *(mPtr + i);
    }
    size_t size() const {
        return mSize;
    }
};

template<typename T>
auto allocBuffer(size_t size) {
    return DataViewer<T>(std::make_shared<Memory>(size * sizeof(T)));
}

namespace {
    template<typename T, typename Deleter>
    struct RAII final {
        const T id;
        RAII(T v) :id(v) {}
        ~RAII() {
            Deleter::destory(id);
        }
    };
}

template<typename T>
CUDA void swap(T& a, T& b) {
    T c = a;
    a = b;
    b = c;
}

