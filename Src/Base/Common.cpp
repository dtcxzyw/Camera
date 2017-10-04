#include <Base/Common.hpp>
#include <system_error>

Memory::Memory(size_t size): mSize(size) {
    checkError(cudaMallocManaged(&mPtr, size));
    checkError(cudaGetDevice(&mDevice));
}

Memory::Memory(const Memory & rhs):Memory(rhs.size()) {
    if (mDevice == rhs.mDevice)
        cudaMemcpy(mPtr, rhs.getPtr(), mSize, cudaMemcpyDefault);
    else
        cudaMemcpyPeer(mPtr, mDevice, rhs.getPtr(), rhs.getDevice(),mSize);
}

Memory::~Memory() {
    checkError(cudaFree(mPtr));
}

char * Memory::getPtr() const {
    return reinterpret_cast<char*>(mPtr);
}

size_t Memory::size() const {
    return mSize;
}

int Memory::getDevice() const {
    return mDevice;
}

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
