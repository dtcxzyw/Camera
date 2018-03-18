#include <Base/Memory.hpp>
#include <vector>
#include <algorithm>
#include <Base/Config.hpp>

struct PinnedMemoryApi final {
    using PtrType = void*;

    static bool alloc(PtrType& ptr, const size_t size) {
        #ifdef CAMERA_MEMORY_MESSAGE
        printf("alloc pinned memory %zu byte\n", size);
        #endif
        return cudaMallocHost(&ptr, size, cudaHostAllocPortable | cudaHostAllocWriteCombined)
            == cudaSuccess;
    }

    static void free(void* ptr) {
        #ifdef CAMERA_MEMORY_MESSAGE
        puts("free pinned memory");
        #endif
        checkError(cudaFreeHost(ptr));
    }
};

struct GlobalMemoryApi final {
    using PtrType = void*;

    static bool alloc(PtrType& res, const size_t size) {
        #ifdef CAMERA_MEMORY_MESSAGE
        printf("alloc device memory %zu byte\n", size);
        #endif
        return cudaMalloc(&res, size) == cudaSuccess;
    }

    static void free(void* ptr) {
        #ifdef CAMERA_MEMORY_MESSAGE
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
    std::vector<typename Api::PtrType> mPool[41];
    uintmax_t mLastRequireTimeStamp[41]{};
    uintmax_t mTimeCount;

    void clearLevel(const size_t level) {
        for (auto&& p : mPool[level])
            Api::free(p);
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

public:
    MemoryPool() : mTimeCount(0) {
        std::fill(std::begin(mLastRequireTimeStamp), std::end(mLastRequireTimeStamp), 0);
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

void clearMemoryPool() {
    getMemoryPool<GlobalMemoryApi>().clear();
    getMemoryPool<PinnedMemoryApi>().clear();
}

GlobalMemoryDeleter::GlobalMemoryDeleter(const size_t size) noexcept: mSize(size) {}

void GlobalMemoryDeleter::operator()(void* ptr) const {
    getMemoryPool<GlobalMemoryApi>().memFree(ptr, mSize);
}

UniqueMemory allocGlobalMemory(const size_t size, const bool isStatic) {
    return UniqueMemory{
        getMemoryPool<GlobalMemoryApi>().memAlloc(size, isStatic),
        GlobalMemoryDeleter(isStatic ? 0 : size)
    };
}

PinnedMemory::PinnedMemory(const size_t size)
    : mPtr(getMemoryPool<PinnedMemoryApi>().memAlloc(size, false)), mSize(size) {}

PinnedMemory::~PinnedMemory() {
    getMemoryPool<PinnedMemoryApi>().memFree(mPtr, mSize);
}

void* PinnedMemory::get() const noexcept {
    return mPtr;
}
