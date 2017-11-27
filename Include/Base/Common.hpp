#pragma once
#include <memory>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cassert>

#define CUDA __device__
#define CUDAInline inline CUDA
#define HOST __host__
#define BOTH CUDA HOST
#define CALLABLE __global__
#define ALIGN __align__(4)
#define ReadOnlyCache __restrict__
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
    T* begin() const {
        return mPtr;
    }
    T* end() const {
        return mPtr + mSize;
    }
    decltype(auto) operator[](size_t i) {
        return *(mPtr + i);
    }
    decltype(auto) operator->() {
        return mPtr;
    }
    decltype(auto) operator*() {
        return *mPtr;
    }
    size_t size() const {
        return mSize;
    }
    void scale(size_t size) {
        mSize = size;
    }
};

template<typename T>
auto allocBuffer(size_t size=1) {
    return DataViewer<T>(std::make_shared<Memory>(size * sizeof(T)));
}

template<typename T>
CUDAInline void swap(T& a, T& b) {
    T c = a;
    a = b;
    b = c;
}

template<typename T>
inline auto calcSize(T a, T b) {
    return (a + b - 1) / b;
}

