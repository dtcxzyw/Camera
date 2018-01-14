#include <Base/Memory.hpp>
#include <vector>

class MemoryPool final :Singletion {
private:
    std::vector<void*> mPool[42];
    MemoryPool() {}
    friend MemoryPool& getMemoryPool();
    void* add(size_t level) {
        auto size = 1 << level;
        void* ptr;
        cudaError_t err;
        while ((err = cudaMallocManaged(&ptr, size)) != cudaSuccess) {
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
        if (size == 0)return nullptr;
        auto level = count(size);
        if (mPool[level].size()) {
            auto ptr = mPool[level].back();
            mPool[level].pop_back();
            return ptr;
        }
        return add(level);
    }

    void memFree(void* ptr, size_t size) {
        if(size)mPool[count(size)].push_back(ptr);
    }

    ~MemoryPool() {
        for (auto&& p : mPool) {
            for (auto&& ptr : p)
                checkError(cudaFree(ptr));
            p.clear();
        }
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
