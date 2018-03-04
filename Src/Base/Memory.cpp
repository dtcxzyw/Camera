#include <Base/Memory.hpp>
#include <vector>
#include <algorithm>

class MemoryPool final : Uncopyable {
private:
    static constexpr auto timeBlock = 1024U;
    std::vector<void*> mPool[41];
    uintmax_t mLastRequireTimeStamp[41]{};
    uintmax_t mTimeCount;

    MemoryPool(): mTimeCount(0) {
        std::fill(std::begin(mLastRequireTimeStamp), std::end(mLastRequireTimeStamp), 0);
    }

    friend MemoryPool& getMemoryPool();

    void clearLevel(const size_t level) {
        for (auto&& p : mPool[level])
            checkError(cudaFree(p));
        mPool[level].clear();
    }

    void gc() {
        auto x = -1;
        for (auto i = 1; i <= 40; ++i)
            if (!mPool[i].empty() && (x == -1 || mLastRequireTimeStamp[i] < mLastRequireTimeStamp[x]))
                x = i;
        if (mTimeCount - mLastRequireTimeStamp[x] > timeBlock)
            clearLevel(x);
    }

    void* tryAlloc(const size_t size) {
        void* ptr;
        cudaError_t err;
        while ((err = cudaMalloc(&ptr, size)) != cudaSuccess) {
            auto flag = true;
            for (auto&& p : mPool) {
                if (!p.empty()) {
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

public:
    void* memAlloc(const size_t size, const bool isStatic) {
        if (size == 0)return nullptr;
        if (isStatic)return tryAlloc(size);
        const auto level = calcSizeLevel(size);
        mLastRequireTimeStamp[level] = ++mTimeCount;
        if (mTimeCount % timeBlock == 0)gc();
        if (!mPool[level].empty()) {
            const auto ptr = mPool[level].back();
            mPool[level].pop_back();
            return ptr;
        }
        return tryAlloc(1ULL<<level);
    }

    void memFree(void* ptr, const size_t size) {
        if (size)mPool[calcSizeLevel(size)].push_back(ptr);
        else checkError(cudaFree(ptr));
    }

    void clear() {
        for (size_t i = 0; i <= 40; ++i)
            clearLevel(i);
    }

    ~MemoryPool() {
        clear();
    }
};

static MemoryPool& getMemoryPool() {
    thread_local static MemoryPool pool;
    return pool;
}

void clearMemoryPool() {
    getMemoryPool().clear();
}

GlobalMemoryDeleter::GlobalMemoryDeleter(const size_t size) noexcept:mSize(size) {}

void GlobalMemoryDeleter::operator()(void * ptr) const{
    getMemoryPool().memFree(ptr,mSize);
}

UniqueMemory allocGlobalMemory(const size_t size, const bool isStatic) {
    return UniqueMemory{ getMemoryPool().memAlloc(size,isStatic),GlobalMemoryDeleter(isStatic?0:size)};
}

PinnedMemory::PinnedMemory(const size_t size): mPtr(nullptr) {
    checkError(cudaMallocHost(&mPtr, size, cudaHostAllocPortable | cudaHostAllocWriteCombined));
}

PinnedMemory::~PinnedMemory() {
    checkError(cudaFreeHost(mPtr));
}

void* PinnedMemory::get() const noexcept {
    return mPtr;
}
