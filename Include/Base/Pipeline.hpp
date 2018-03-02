#pragma once
#include <Base/Common.hpp>
#include <Base/Memory.hpp>
#include <Base/Math.hpp>
#include <Base/Config.hpp>

template<typename Func, typename... Args>
CUDAINLINE void run(Func func,unsigned int block, unsigned int size, Args... args) {
    if (size)func << <calcSize(size, block), min(block, size)>> > (size, args...);
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
    void run(Func func, unsigned int size, Args... args) {
        if (size) {
            func <<<calcSize(size, mMaxThread),min(mMaxThread,size), 0, mStream >>> (size, args...);
            checkError();
#ifdef CAMERA_LAUNCH_SYNC
            sync();
#endif
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
    void memset(DataViewer<T> data, const int val=0) {
        checkError(cudaMemsetAsync(data.begin(),val,data.size()*sizeof(T),mStream));
    }

    void wait(Event& event);
};

class Event final :Uncopyable {
private:
    cudaEvent_t mEvent{};
public:
    explicit Event(Stream& stream);
    void wait();
    cudaEvent_t get() const;
    ~Event();
};

using SharedEvent = std::shared_ptr<Event>;

