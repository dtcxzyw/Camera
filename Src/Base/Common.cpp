#include <Base/Common.hpp>
#include <system_error>

Memory::Memory(size_t size): mSize(size) {
    checkError(cudaMallocManaged(&mPtr, size));
    checkError(cudaMemAdvise(mPtr,size,cudaMemAdviseSetReadMostly,getDevice().getId()));
}

Memory::Memory(const Memory & rhs):Memory(rhs.size()) {
    cudaMemcpy(mPtr, rhs.getPtr(), rhs.size(),cudaMemcpyDefault);
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

Device& getDevice() {
    static Device device;
    return device;
}

void checkError(cudaError_t error) {
    if (error != cudaSuccess) {
        puts(cudaGetErrorName(error));
        puts(cudaGetErrorString(error));
        throw std::exception(cudaGetErrorString(error));
    }
}

void checkError() {
    checkError(cudaGetLastError());
}

std::vector<cudaDeviceProp> Device::enumDevices() {
    std::vector<cudaDeviceProp> res;
    int count;
    checkError(cudaGetDeviceCount(&count));
    res.resize(count);
    for (int i = 0; i < count; ++i)
        checkError(cudaGetDeviceProperties(&res[i],i));
    return res;
}

void Device::init(int id) {
    mCurrentId = id;
    checkError(cudaSetDevice(id));
    checkError(cudaGetDeviceProperties(&mProp,id));
}

void Device::sync() {
    checkError(cudaDeviceSynchronize());
}

Device::~Device() {
    checkError(cudaDeviceReset());
}

const cudaDeviceProp & Device::getProp() {
    return mProp;
}

int Device::getId() {
    return mCurrentId;
}
