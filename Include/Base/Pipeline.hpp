#pragma once
#include "Common.hpp"
#include <thread>
#include <future>
#include <functional>
#include <map>

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

class GPUInstance;

class Environment final :Singletion {
private:
    Environment();
    friend Environment& getEnvironment();
    std::vector<std::thread> mThreads;
    std::map<int,GPUInstance*> mDevices;
    bool mFlag;
public:
    void init();
    std::future<void> pushTask(std::function<void()>&& deferred);
    const cudaDeviceProp& getProp() const;
    ~Environment();
};

Environment& getEnvironment();

