#pragma once
#include "Common.hpp"

CUDAInline unsigned int getID() {
    return blockIdx.x*blockDim.x + threadIdx.x;
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

    template<typename Func, typename... Args>
    void run(Func func, unsigned int size, Args... args) {
        if (size) {
            func <<<calcSize(size, mMaxThread), glm::min(mMaxThread, size), 0, mStream >>> (size, args...);
            checkError();
        }
    }

    template<typename Func, typename... Args>
    void runDim(Func func, dim3 grid, dim3 block, Args... args) {
        func <<<grid, block,0, mStream >>> (args...);
        checkError();
    }

    unsigned int getMaxBlockSize() const {
        return mMaxThread;
    }

    template<typename T>
    auto share(const T* data, size_t size) {
        auto rsize = size * sizeof(T);
        auto sm = std::make_shared<Memory>(rsize);
        checkError(cudaMemcpyAsync(sm->getPtr(), data, rsize, cudaMemcpyDefault,mStream));
        return DataViewer<T>(sm);
    }

    template<typename C>
    auto share(const C& c) {
        using T = std::decay_t<decltype(*c.data()) >;
        return share(c.data(), c.size());
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

