#include <Core/DeviceMemory.hpp>
//TODO: memory pool

CUDA DeviceMemoryDesc alloc(const unsigned int size) {
    DeviceMemoryDesc res;
    res.ptr = malloc(size);
    res.size = size;
    return res;
}

CUDA void free(const DeviceMemoryDesc desc) {
    free(desc.ptr);
}
