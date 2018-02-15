#include <Base/Common.hpp>
#include <cstdio>
#include <stdexcept>

void checkError(const cudaError_t error) {
    if (error != cudaSuccess) {
        puts(cudaGetErrorName(error));
        puts(cudaGetErrorString(error));
        __debugbreak();
        throw std::runtime_error(cudaGetErrorString(error));
    }
    //reset error
    cudaGetLastError();
}

void checkError() {
    checkError(cudaGetLastError());
}
