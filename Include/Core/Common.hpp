#pragma once
#include <Core/Config.hpp>
#include <Core/IncludeBegin.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Core/IncludeEnd.hpp>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

#define DEVICE __device__
#define DEVICEINLINE inline DEVICE
#define HOST __host__
#define BOTH DEVICE HOST
#define GLOBAL __global__
#define ALIGN __align__(TRANSFER_ALIGN)
#define STRUCT_ALIGN __align__(CACHE_ALIGN)
#define READONLY(type) const type* __restrict__
#define NOT_IMPLEMENTED() throw std::logic_error("Not implemented.")
#define LAUNCH_BOUND __launch_bounds__

#ifdef CAMERA_DEBUG
#define CHECKFP(expr) \
    {\
        const auto val=(expr);\
        if (!isfinite(val)) {\
            printf("floating point check failed (value = %f) at %s : %s line %d\n", val, __FILE__, __FUNCTION__, __LINE__); \
        }\
    }
#else
#define CHECKFP(expr)
#endif

struct Uncopyable {
    Uncopyable() = default;
    ~Uncopyable() = default;
    Uncopyable(const Uncopyable&) = delete;
    Uncopyable(Uncopyable&&) = default;
    Uncopyable& operator=(const Uncopyable&) = delete;
    Uncopyable& operator=(Uncopyable&&) = default;
};

template <typename T>
class Singletion {
protected:
    Singletion() = default;
public:
    Singletion(const Singletion&) = delete;
    Singletion(Singletion&&) = delete;
    Singletion& operator=(const Singletion&) = delete;
    Singletion& operator=(Singletion&&) = delete;
    ~Singletion() = default;

    static T& get() {
        static T singletion;
        return singletion;
    }
};

void debugBreak();
void checkError(cudaError_t error);
void checkError();

template <typename T>
BOTH auto calcBlockSize(const T a, const T b) {
    return (a + b - 1) / b;
}

DEVICEINLINE uint32_t getId() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

template <typename T>
DEVICEINLINE void cudaSwap(T& a, T& b) {
    auto c = a;
    a = b;
    b = c;
}

struct Empty final {};
