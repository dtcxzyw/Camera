#pragma once
#include "Common.hpp"

constexpr auto blockSize = 1024U;

inline CUDA uint getID() {
    return blockIdx.x*blockSize + threadIdx.x;
}

inline uint calcSize(uint a,uint b) {
    return (a+b-1) / b;
}

class Pipeline final {
private:
    cudaStream_t mStream;
public:
    Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    ~Pipeline();

    void sync();
    cudaStream_t getId() const;

    template<cudaFuncCache cache = cudaFuncCachePreferL1, typename Func, typename... Args>
    void run(Func func, unsigned int size, Args... args) {
        checkError(cudaFuncSetCacheConfig(func, cache));
        func <<<calcSize(size,blockSize), glm::min(blockSize, size),0, mStream>>> (size, args...);
        checkError();
    }

    template<cudaFuncCache cache=cudaFuncCachePreferL1,typename Func, typename... Args>
    void runDim(Func func, dim3 grid, dim3 block, Args... args) {
        checkError(cudaFuncSetCacheConfig(func, cache));
        func <<<grid, block,0, mStream >>> (args...);
        checkError();
    }
};
