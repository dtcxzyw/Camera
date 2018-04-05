#pragma once
#include <memory>
#include <Base/Common.hpp>
#include <Base/Math.hpp>

inline size_t calcSizeLevel(const size_t size) {
    const auto msb=findMSB(size);
    return msb + (size != (1ULL << msb));
}

template<typename T>
size_t calcMaxBufferSize(const size_t size) {
    const auto level = calcSizeLevel(size*sizeof(T));
    return (1 << level) / sizeof(T);
}

void clearMemoryPool();

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

    explicit DataViewer(UniqueMemory memory, const size_t size)
        : mMem(std::move(memory)), mSize(size) {}

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
    size_t mSize;
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
