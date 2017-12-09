#pragma once
#include "Common.hpp"
#include <functional>
#include <deque>

CUDAInline unsigned int getID() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

class Event;

class Stream final:Uncopyable {
private:
    cudaStream_t mStream;
    unsigned int mMaxThread;
public:
    Stream();
    ~Stream();

    void sync();
    cudaStream_t getID() const;
    cudaError_t query() const;

    template<typename Func, typename... Args>
    void run(Func func, unsigned int size, Args... args) {
        if (size) {
            func <<<calcSize(size, mMaxThread),min(mMaxThread,size), 0, mStream >>> (size, args...);
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
        using T = std::decay_t<decltype(*std::data(c)) >;
        return share(std::data(c), std::size(c));
    }

    template<typename T>
    void memset(DataViewer<T> data,int val=0) {
        checkError(cudaMemsetAsync(data.begin(),val,data.size()*sizeof(T),mStream));
    }

    template<typename T>
    auto copy(DataViewer<T> data) {
        return share(data.begin(), data.size());
    }

    void wait(Event& event);
};

class Event final :Uncopyable {
private:
    cudaEvent_t mEvent;
public:
    Event(Stream& stream);
    void wait();
    cudaEvent_t get();
    ~Event();
};

using SharedEvent = std::shared_ptr<Event>;

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

