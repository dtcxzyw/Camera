#include <Base/Common.hpp>
#include <cstdio>
#include <stdexcept>

void debugBreak() {
#if defined CAMERA_DEBUG && defined _WIN32
    __debugbreak();
#endif
}

void checkError(const cudaError_t error) {
    if (error != cudaSuccess) {
        //reset error
        cudaGetLastError();
        puts(cudaGetErrorName(error));
        puts(cudaGetErrorString(error));
        debugBreak();
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void checkError() {
    checkError(cudaGetLastError());
}
