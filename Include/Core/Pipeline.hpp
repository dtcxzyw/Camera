#pragma once
#include <Core/Common.hpp>
#include <Core/Memory.hpp>
#include <Math/Math.hpp>

template<typename Func, typename... Args>
CUDAINLINE void launchLinear(Func func,unsigned int block, unsigned int size, Args... args) {
    if (size)func << <calcBlockSize(size, block), min(block, size)>> > (size, args...);
}

class Event;

class Stream final:Uncopyable {
private:
    cudaStream_t mStream{};
    unsigned int mMaxThread;
public:
    Stream();
    ~Stream();

    void sync();
    cudaStream_t get() const;
    cudaError_t query() const;

    template<typename Func, typename... Args>
    void launchLinear(Func func, unsigned int size, Args... args) {
        if (size) {
            func <<<calcBlockSize(size, mMaxThread),min(mMaxThread,size), 0, mStream >>> (size, args...);
            checkError();
        }
    }

    template<typename Func, typename... Args>
    void launchDim(Func func, dim3 grid, dim3 block, Args... args) {
        func <<<grid, block,0, mStream >>> (args...);
        checkError();
    }

    unsigned int getMaxBlockSize() const {
        return mMaxThread;
    }

    template<typename T>
    void memset(const MemorySpan<T>& data, const int val=0) {
        checkError(cudaMemsetAsync(data.begin(),val,data.size()*sizeof(T),mStream));
    }

    void wait(Event& event);
};

class Event final :Uncopyable {
private:
    cudaEvent_t mEvent{};
public:
    explicit Event(bool recordTime = false);
    void bind(Stream& stream);
    void sync();
    cudaError_t query() const;
    cudaEvent_t get() const;
    float operator-(Event& rhs) const;
    ~Event();
};

using SharedEvent = std::shared_ptr<Event>;

