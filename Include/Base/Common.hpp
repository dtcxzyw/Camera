#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA __device__
#define CUDAInline __forceinline__ CUDA
#define HOST __host__
#define BOTH CUDA HOST
#define CALLABLE __global__
#define ALIGN __align__(4)
#define ReadOnlyCache __restrict__
#include "Math.hpp"

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

template<typename T>
BOTH auto calcSize(T a, T b) {
    return (a + b - 1) / b;
}

CUDAInline unsigned int getID() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

