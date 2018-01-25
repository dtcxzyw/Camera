#include <Base/Memory.hpp>
#include <vector>
#include <algorithm>

class MemoryPool final :Singletion {
private:
    static constexpr auto timeBlock = 1024U;
    std::vector<void*> mPool[41];
    uintmax_t mLastRequireTimeStamp[41];
    uintmax_t mTimeCount;
    MemoryPool():mTimeCount(0) {
        std::fill(std::begin(mLastRequireTimeStamp),std::end(mLastRequireTimeStamp),0);
    }
    friend MemoryPool& getMemoryPool();
    void clearLevel(std::vector<void*>& level) {
        for (auto&& p : level)
            checkError(cudaFree(p));
        level.clear();
    }
    void GC() {
        int x=-1;
        for (auto i = 1; i <= 40; ++i)
            if (!mPool[i].empty() && (x == -1 || mLastRequireTimeStamp[i] < mLastRequireTimeStamp[x]))
                x = i;
        if (mTimeCount - mLastRequireTimeStamp[x] > timeBlock)
            clearLevel(mPool[x]);
    }
    void* add(size_t level) {
        auto size = 1 << level;
        void* ptr;
        cudaError_t err;
        while ((err = cudaMalloc(&ptr, size)) != cudaSuccess) {
            auto flag = true;
            for (auto&& p : mPool) {
                if (p.size()) {
                    flag = false;
                    checkError(cudaFree(p.back()));
                    p.pop_back();
                    break;
                }
            }
            if (flag) checkError(err);
        }
        return ptr;
    }
    size_t count(size_t x) {
        for (int i = 40; i >= 0; --i)
            if (x&(1ULL << i))
                return i + 1;
        return -1;
    }
public:
    void* memAlloc(size_t size) {
        auto level = count(size);
        mLastRequireTimeStamp[level] = ++mTimeCount;
        if(mTimeCount%timeBlock==0)GC();
        if (mPool[level].size()) {
            auto ptr = mPool[level].back();
            mPool[level].pop_back();
            return ptr;
        }
        return add(level);
    }

    void memFree(void* ptr, size_t size) {
        mPool[count(size)].push_back(ptr);
    }

    ~MemoryPool() {
        for (auto&& p : mPool)
            clearLevel(p);
    }
};

MemoryPool& getMemoryPool() {
    thread_local static MemoryPool pool;
    return pool;
}

Memory::Memory(size_t size) : mSize(size), mPtr(getMemoryPool().memAlloc(size)) {}

Memory::~Memory() {
    getMemoryPool().memFree(mPtr, mSize);
}

char * Memory::getPtr() const {
    return reinterpret_cast<char*>(mPtr);
}

size_t Memory::size() const {
    return mSize;
}
