#include <Base/Pipeline.hpp>
#include <queue>
#include <mutex>
#include <atomic>
#include <set>
using namespace std::chrono_literals;

Pipeline::Pipeline() {
    checkError(cudaStreamCreate(&mStream));
    mMaxThread = getEnvironment().getProp().maxThreadsPerBlock;
}

Pipeline::~Pipeline() {
    checkError(cudaStreamDestroy(mStream));
}

void Pipeline::sync() {
    checkError(cudaStreamSynchronize(mStream));
}

cudaStream_t Pipeline::getId() const {
    return mStream;
}

static std::atomic_size_t cnt;
static std::set<size_t> table;
static std::mutex tableMutex;

class GPUInstance final :Uncopyable {
private:
    cudaDeviceProp mProp;
    std::queue<std::function<void()>> mQueue;
    std::mutex mQueueMutex;
public:
    GPUInstance(int id) {
        checkError(cudaSetDevice(id));
        checkError(cudaGetDeviceProperties(&mProp, id));
    }
    const cudaDeviceProp& getProp() const {
        return mProp;
    }
    std::future<void> insert(std::function<void()>&& deferred) {
        {
            std::lock_guard<std::mutex> guard(mQueueMutex);
            mQueue.emplace(std::move(deferred));
        }
        auto id = ++cnt;
        return std::async(std::launch::deferred, [id] {
            while (!table.count(id))std::this_thread::yield();
            std::lock_guard<std::mutex> guard(tableMutex);
            table.erase(id);
        });
    }
    size_t size() const {
        return mQueue.size();
    }
    void update() {
        std::function<void()> func;
        {
            std::lock_guard<std::mutex> guard(mQueueMutex);
            if (mQueue.empty()) {
                std::this_thread::sleep_for(10us);
                return;
            }
            func = std::move(mQueue.front());
            mQueue.pop();
        }
        func();
    }
    ~GPUInstance() {
        checkError(cudaDeviceReset());
    }
};

Environment::Environment() :mFlag(true) {}

void Environment::init() {
    int cnt;
    checkError(cudaGetDeviceCount(&cnt));
    cudaDeviceProp prop;
    for (int i = 0; i < cnt; ++i) {
        checkError(cudaGetDeviceProperties(&prop, i));
        int ver = prop.major * 10000 + prop.minor;
        if (ver < 50002)continue;
        if (!prop.unifiedAddressing)continue;
        if (!prop.managedMemory)continue;
        mThreads.emplace_back([this, i] {
            GPUInstance device(i);
            mDevices[i] = &device;
            while (mFlag) device.update();
        });
    }
    while (mDevices.size() != mThreads.size())
        std::this_thread::yield();
    if (mDevices.empty())
        throw std::exception("Failed to initalize the CUDA environment.");
}

std::future<void> Environment::pushTask(std::function<void()>&& deferred) {
    auto dev = std::min_element(mDevices.begin(), mDevices.end(), [](auto a, auto b) {
        return a.second->size() < b.second->size();
    });
    return dev->second->insert(std::move(deferred));
}

const cudaDeviceProp& Environment::getProp() const {
    int id;
    checkError(cudaGetDevice(&id));
    return mDevices.find(id)->second->getProp();
}

Environment::~Environment() {
    mFlag = false;
    for (auto&& t : mThreads)
        if (t.joinable())
            t.join();
}

Environment& getEnvironment() {
    static Environment env;
    return env;
}
