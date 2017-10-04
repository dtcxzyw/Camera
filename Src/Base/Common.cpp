#include <Base/Common.hpp>
#include <Base/Pipeline.hpp>
#include <system_error>
#include <map>
#include <mutex>
#include <queue>

class MemoryPool final:Singletion {
private:
    std::multimap<size_t, void*> mPool;
    std::queue<size_t> mQueue;
    size_t mSize;
    std::mutex mPoolMutex;
    MemoryPool() :mSize(0) {}
    friend MemoryPool& getMemoryPool();
    void release(decltype(mPool)::iterator p) {
        mSize -= p->first;
        checkError(cudaFree(p->second));
        mPool.erase(p);
    }
    void blance() {
        if (mQueue.size()) {
            auto s = mQueue.front();
            auto p = mPool.find(s);
            if (p != mPool.end())release(p);
            mQueue.pop();
        }
    }
public:

    void* memAlloc(size_t size) {
        std::lock_guard<std::mutex> guard(mPoolMutex);
        auto p = mPool.upper_bound(size-1);
        if (p != mPool.cend()) {
            auto ptr = p->second;
            mPool.erase(p);
            return ptr;
        }
        else {
            void** ptr;
            cudaError_t error;
            while ((error=cudaMallocManaged(&ptr, size)) != cudaSuccess) {
                if (mPool.size()) release(mPool.begin());
                else checkError(error);
            }
            mSize += size;
            return ptr;
        }
    }

    void memFree(void* ptr, size_t size) {
        std::lock_guard<std::mutex> guard(mPoolMutex);
        mPool.emplace(size, ptr);
        while (mSize >= (1LL << 30))blance();
        mQueue.push(size);
    }

    ~MemoryPool() {
        for (auto&& x : mPool)
            checkError(cudaFree(x.second));
    }
};

MemoryPool& getMemoryPool() {
    static MemoryPool pool;
    return pool;
}

Memory::Memory(size_t size): mSize(size),mPtr(getMemoryPool().memAlloc(size)) {}

Memory::~Memory() {
    getMemoryPool().memFree(mPtr, mSize);
}

char * Memory::getPtr() const {
    return reinterpret_cast<char*>(mPtr);
}

size_t Memory::size() const {
    return mSize;
}

void checkError(cudaError_t error) {
    if (error != cudaSuccess) {
        puts(cudaGetErrorName(error));
        puts(cudaGetErrorString(error));
        __debugbreak();
        throw std::exception(cudaGetErrorString(error));
    }
}

void checkError() {
    checkError(cudaGetLastError());
}
