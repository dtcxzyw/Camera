#include <Core/DeviceMemory.hpp>
//TODO: memory pool

DEVICE DeviceMemoryDesc alloc(const uint32_t size) {
    DeviceMemoryDesc res;
    res.ptr = malloc(size);
    res.size = size;
    return res;
}

DEVICE void free(const DeviceMemoryDesc desc) {
    free(desc.ptr);
}
