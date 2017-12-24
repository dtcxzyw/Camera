#include <Base/Common.hpp>
#include <exception>
#include <cstdio>

void checkError(cudaError_t error) {
    if (error != cudaSuccess) {
        puts(cudaGetErrorName(error));
        puts(cudaGetErrorString(error));
        __debugbreak();
        throw std::exception(cudaGetErrorString(error));
    }
}

void checkError() {
    checkError(cudaGetLastError());
}
