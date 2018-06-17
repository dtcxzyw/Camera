#pragma once
#include <Core/Common.hpp>
#include <Core/IncludeBegin.hpp>
#include <device_atomic_functions.h>
#include <sm_20_atomic_functions.h>
#include <device_functions.h>
#include <math_functions.h>
#include <Core/IncludeEnd.hpp>

template <typename T>
DEVICEINLINE T deviceAtomicAdd(T* address, const T val) {
    return atomicAdd(address, val);
}

template <typename T>
DEVICEINLINE T deviceAtomicInc(T* address, const T maxv) {
    return atomicInc(address, maxv);
}

template <typename T>
DEVICEINLINE T deviceAtomicMin(T* address, const T val) {
    return atomicMin(address, val);
}

template <typename T>
DEVICEINLINE T deviceCompareAndSwap(T* address, const T comp, const T val) {
    return atomicCAS(address, comp, val);
}

template <typename T>
DEVICEINLINE T deviceSaturate(const T val) {
    return saturate(val);
}
