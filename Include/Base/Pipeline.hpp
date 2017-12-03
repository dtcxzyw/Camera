#pragma once
#include "Common.hpp"
#include <functional>
#include <deque>

CUDAInline unsigned int getID() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

constexpr auto maxThread = 1024U;

template<typename Func, typename... Args>
CUDAInline void run(Func func, unsigned int size, Args... args) {
    if (size) func << <calcSize(size, maxThread), min(maxThread,size) >> > (size, args...);
}

template<typename Func, typename... Args>
CUDAInline void runDim(Func func, dim3 grid, dim3 block, Args... args) {
    func <<<grid, block>>> (args...);
}

class Event;

class Stream final:Uncopyable {
private:
    cudaStream_t mStream;
    unsigned int mMaxThread;
public:
    Stream();
    Stream(Stream&& rhs);
    Stream& operator=(Stream&& rhs);
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

template<typename Task>
class Pipeline final :Uncopyable {
private:
    std::deque<std::pair<Task,SharedEvent>> mTasks;
    std::vector<std::function<void(Task&, Stream&)>> mStages;
    std::vector<Stream> mStreams;
    std::function<Task()> mBegin;
    std::function<void(Task&)> mEnd;
public:
    Pipeline(std::function<Task()> begin):mBegin(std::move(begin)){}
    Pipeline(Pipeline&&) = default;
    Pipeline& operator=(Pipeline&&) = default;
    Pipeline& then(std::function<void(Task&, Stream&)> stage) {
        if(!mStreams.empty())throw std::exception("This pipeline is complete.");
        mStages.emplace_back(std::move(stage));
        return *this;
    }
    Pipeline& end(std::function<void(Task&)> end) {
        if (!mStreams.empty())throw std::exception("This pipeline is complete.");
        mEnd = std::move(end);
        mStreams.resize(mStages.size());
        return *this;
    }
    void update() {
        if (mStreams.empty())throw std::exception("This pipeline is incomplete.");
        if (mTasks.size() == mStreams.size()) {
            mTasks.back().second->wait();
            mEnd(mTasks.back().first);
            mTasks.pop_back();
        }
        unsigned int idx = 0;
        for (auto&& task:mTasks) {
            ++idx;
            mStreams[idx].wait(*task.second);
            mStages[idx](task.first, mStreams[idx]);
            task.second = std::make_shared<Event>(mStreams[idx]);
        }
        mTasks.emplace_front(mBegin(),nullptr);
        mStages.front()(mTasks.front().first,mStreams.front());
        mTasks.front().second=std::make_shared<Event>(mStreams.front());
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

