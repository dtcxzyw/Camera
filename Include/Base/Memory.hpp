#pragma once
#include <memory>
#include <Base/Common.hpp>

inline size_t calcSizeLevel(const size_t size) {
    for (auto i = 40; i >= 0; --i)
        if (size & (1ULL << i))
            return i + 1;
    return -1;
}

template<typename T>
size_t calcMaxBufferSize(const size_t size) {
    auto level = calcSizeLevel(size*sizeof(T));
    return (1 << level) / sizeof(T);
}

class GlobalMemoryDeleter final {
private:
    size_t mSize;
public:
    constexpr GlobalMemoryDeleter() noexcept:mSize(0){}
    explicit GlobalMemoryDeleter(size_t size) noexcept;
    void operator()(void* ptr) const;
};

using UniqueMemory = std::unique_ptr<void,GlobalMemoryDeleter>;

UniqueMemory allocGlobalMemory(size_t size,bool isStatic=false);

template <typename T>
class DataViewer final {
private:
    std::shared_ptr<void> mMem;
    size_t mSize;
public:
    DataViewer() :mSize(0){}
    explicit DataViewer(const size_t size)
        : mMem(allocGlobalMemory(size*sizeof(T),true)),mSize(size) {}

    T* begin() const {
        return reinterpret_cast<T*>(mMem.get());
    }

    T* end() const {
        return begin() + mSize;
    }

    size_t size() const {
        return mSize;
    }
};

class PinnedMemory final : Uncopyable {
private:
    void* mPtr;
public:
    explicit PinnedMemory(size_t size);
    ~PinnedMemory();
    void* get() const noexcept;
};

template <typename T>
class PinnedBuffer final : Uncopyable {
private:
    PinnedMemory mMemory;
public:
    explicit PinnedBuffer(const size_t size) : mMemory(size * sizeof(T)) {}

    T* get() const noexcept {
        return reinterpret_cast<T*>(mMemory.get());
    }

    T& operator[](const size_t idx) {
        return get()[idx];
    }

    T& operator*() {
        return *get();
    }

    T operator*() const{
        return *get();
    }
};
