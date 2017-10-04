#include <Base/Common.hpp>
#include <Base/Pipeline.hpp>
#include <system_error>
#include <map>
#include <mutex>
#include <list>

class MemoryPool final:Singletion {
private:
    std::multimap<size_t,void*> mPool;
    std::mutex mPoolMutex;
    MemoryPool() {}
    friend MemoryPool& getMemoryPool();
    void* add(size_t size) {
        void* ptr;
        cudaError_t err;
        while ((err=cudaMallocManaged(&ptr, size))!=cudaSuccess) {
            if (mPool.size()) {
                checkError(cudaFree(mPool.begin()->second));
                mPool.erase(mPool.begin());
            }
            else checkError(err);
        }
        return ptr;
    }
    size_t count(size_t x) {
        for (int i = 40; i >= 0; --i)
            if (x&(1ULL << i))
                return 1ULL<<(i+1);
        return -1;
    }
public:
    void* memAlloc(size_t size) {
        auto level = count(size);
        std::lock_guard<std::mutex> guard(mPoolMutex);
        auto p = mPool.upper_bound(size-1);
        if (p != mPool.cend()) {
            auto ptr = p->second;
            mPool.erase(p);
            return ptr;
        }
        return add(level);
    }

    void memFree(void* ptr, size_t size) {
        std::lock_guard<std::mutex> guard(mPoolMutex);
        mPool.emplace(count(size), ptr);
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
