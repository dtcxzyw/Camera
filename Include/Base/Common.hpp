#pragma once
#include <Base/CompileBegin.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Base/CompileEnd.hpp>

#define CUDA __device__
#define CUDAINLINE __forceinline__ CUDA
#define CUDANOINLINE __noinline__ CUDA
#define HOST __host__
#define BOTH CUDAINLINE HOST
#define GLOBAL __global__
#define ALIGN __align__(16)
#define READONLY(type) const type* __restrict__ const

struct Uncopyable {
    Uncopyable() = default;
    ~Uncopyable() = default;
    Uncopyable(const Uncopyable&) = delete;
    Uncopyable(Uncopyable&&) = default;
    Uncopyable& operator=(const Uncopyable&) = delete;
    Uncopyable& operator=(Uncopyable&&) = default;
};

class Singletion {
protected:
    Singletion() = default;
    ~Singletion() = default;
public:
    Singletion(const Singletion&) = delete;
    Singletion(Singletion&&) = delete;
    Singletion& operator=(const Singletion&) = delete;
    Singletion& operator=(Singletion&&) = delete;
};

void checkError(cudaError_t error);
void checkError();

template <typename T>
BOTH auto calcSize(const T a,const T b) {
    return (a + b - 1) / b;
}

CUDAINLINE unsigned int getID() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template <typename T>
CUDAINLINE void cudaSwap(T& a, T& b) {
    auto c = a;
    a = b;
    b = c;
}
