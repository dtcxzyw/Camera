#pragma once
#include "Common.hpp"

inline CUDA uint getID() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

inline uint calcSize(uint a,uint b) {
    return (a+b-1) / b;
}

class Pipeline final:Uncopyable {
private:
    cudaStream_t mStream;
    unsigned int mMaxThread;
public:
    Pipeline();
    ~Pipeline();

    void sync();
    cudaStream_t getId() const;

    template<cudaFuncCache cache = cudaFuncCachePreferL1, typename Func, typename... Args>
    void run(Func func, unsigned int size, Args... args) {
        if (size) {
            checkError(cudaFuncSetCacheConfig(func, cache));
            func <<<calcSize(size, mMaxThread), glm::min(mMaxThread, size), 0, mStream >>> (size, args...);
            checkError();
        }
    }

    template<cudaFuncCache cache=cudaFuncCachePreferL1,typename Func, typename... Args>
    void runDim(Func func, dim3 grid, dim3 block, Args... args) {
        checkError(cudaFuncSetCacheConfig(func, cache));
        func <<<grid, block,0, mStream >>> (args...);
        checkError();
    }
};

class Environment final :Singletion {
private:
    Environment();
    friend Environment& getEnvironment();
    cudaDeviceProp mProp;
public:
    void init();
    const cudaDeviceProp& getProp() const;
    ~Environment();
};

Environment& getEnvironment();

