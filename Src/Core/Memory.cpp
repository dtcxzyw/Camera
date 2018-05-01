#include <Core/Common.hpp>
#include <Core/Memory.hpp>
#include <vector>
#include <algorithm>
#ifdef CAMERA_MEMORY_TRACE
#include <set>
#endif

struct PinnedMemoryApi final {
    using PtrType = void*;

    static bool alloc(PtrType& ptr, const size_t size) {
        #ifdef CAMERA_MEMORY_TRACE
        printf("alloc pinned memory %zu byte\n", size);
        #endif
        return cudaMallocHost(&ptr, size, cudaHostAllocPortable | cudaHostAllocWriteCombined)
            == cudaSuccess;
    }

    static void free(void* ptr) {
        #ifdef CAMERA_MEMORY_TRACE
        puts("free pinned memory");
        #endif
        checkError(cudaFreeHost(ptr));
    }
};

struct GlobalMemoryApi final {
    using PtrType = void*;

    static bool alloc(PtrType& res, const size_t size) {
        #ifdef CAMERA_MEMORY_TRACE
        printf("alloc device memory %zu byte\n", size);
        #endif
        return cudaMalloc(&res, size) == cudaSuccess;
    }

    static void free(void* ptr) {
        #ifdef CAMERA_MEMORY_TRACE
        puts("free device memory");
        #endif
        checkError(cudaFree(ptr));
    }
};

template <typename Api>
class MemoryPool final : Uncopyable {
public:
    using PtrType = typename Api::PtrType;
private:
    static constexpr auto timeBlock = 1024U;
    std::vector<PtrType> mPool[41];
    uintmax_t mLastRequireTimeStamp[41]{};
    uintmax_t mTimeCount;

    #ifdef CAMERA_MEMORY_TRACE
    std::set<PtrType> mAlloced;
    #endif

    void clearLevel(const size_t level) {
        for (auto&& p : mPool[level])
            Api::free(p);
        mPool[level].clear();
    }

    PtrType tryAlloc(const size_t size) {
        PtrType ptr;
        while (!Api::alloc(ptr, size)) {
            auto flag = true;
            for (auto&& p : mPool) {
                if (!p.empty()) {
                    flag = false;
                    Api::free(p.back());
                    p.pop_back();
                    break;
                }
            }
            if (flag) throw std::bad_alloc{};
        }
        return ptr;
    }

    PtrType memAlloc(const size_t size, const bool isStatic) {
        if (size == 0)return {};
        if (isStatic)return tryAlloc(size);
        const auto level = calcSizeLevel(size);
        mLastRequireTimeStamp[level] = ++mTimeCount;
        if (mTimeCount % timeBlock == 0)gc();
        if (!mPool[level].empty()) {
            const auto ptr = mPool[level].back();
            mPool[level].pop_back();
            return ptr;
        }
        return tryAlloc(1ULL << level);
    }

    void memFree(PtrType ptr, const size_t size) {
        if (size)mPool[calcSizeLevel(size)].push_back(ptr);
        else checkError(cudaFree(ptr));
    }

public:
    MemoryPool() : mTimeCount(0) {
        std::fill(std::begin(mLastRequireTimeStamp), std::end(mLastRequireTimeStamp), 0);
    }

    void gc() {
        auto x = 0;
        for (auto i = 1; i <= 40; ++i)
            if (!mPool[i].empty() && 
                (mPool[x].empty() || mLastRequireTimeStamp[i] < mLastRequireTimeStamp[x]))
                x = i;
        if (!mPool[x].empty() && mTimeCount - mLastRequireTimeStamp[x] > timeBlock)
            clearLevel(x);
    }

    PtrType alloc(const size_t size, const bool isStatic) {
        const auto ptr = memAlloc(size, isStatic);
        #ifdef CAMERA_MEMORY_TRACE
        mAlloced.emplace(ptr);
        #endif
        return ptr;
    }

    void free(PtrType ptr, const size_t size) {
        #ifdef CAMERA_MEMORY_TRACE
        if (!mAlloced.erase(ptr))throw std::logic_error("freed ptr");
        #endif
        memFree(ptr, size);
    }

    void clear() {
        for (size_t i = 0; i <= 40; ++i)
            clearLevel(i);
    }

    ~MemoryPool() {
        clear();
    }
};

template <typename Api>
static MemoryPool<Api>& getMemoryPool() {
    thread_local static MemoryPool<Api> pool;
    return pool;
}

void gc(){
    getMemoryPool<GlobalMemoryApi>().gc();
    getMemoryPool<PinnedMemoryApi>().gc();
}

void clearMemoryPool() {
    getMemoryPool<GlobalMemoryApi>().clear();
    getMemoryPool<PinnedMemoryApi>().clear();
}

GlobalMemoryDeleter::GlobalMemoryDeleter(const size_t size) noexcept: mSize(size) {}

void GlobalMemoryDeleter::operator()(void* ptr) const {
    getMemoryPool<GlobalMemoryApi>().free(ptr, mSize);
}

UniqueMemory allocGlobalMemory(const size_t size, const bool isStatic) {
    return UniqueMemory{
        getMemoryPool<GlobalMemoryApi>().alloc(size, isStatic),
        GlobalMemoryDeleter(isStatic ? 0 : size)
    };
}

PinnedMemory::PinnedMemory(const size_t size)
    : mPtr(getMemoryPool<PinnedMemoryApi>().alloc(size, false)), mSize(size) {}

PinnedMemory::~PinnedMemory() {
    getMemoryPool<PinnedMemoryApi>().free(mPtr, mSize);
}

void* PinnedMemory::get() const noexcept {
    return mPtr;
}
