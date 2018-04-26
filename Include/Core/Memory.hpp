#pragma once
#include <memory>
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#ifdef CAMERA_DEBUG
#include <stdexcept>
#endif

inline size_t calcSizeLevel(const size_t size) {
    const auto msb = glm::findMSB(size);
    return msb + (size != (1ULL << msb));
}

template <typename T>
size_t calcMaxBufferSize(const size_t size) {
    const auto level = calcSizeLevel(size * sizeof(T));
    return (1 << level) / sizeof(T);
}

void clearMemoryPool();

class GlobalMemoryDeleter final {
private:
    size_t mSize;
public:
    constexpr GlobalMemoryDeleter() noexcept: mSize(0) {}
    explicit GlobalMemoryDeleter(size_t size) noexcept;
    void operator()(void* ptr) const;
};

using UniqueMemory = std::unique_ptr<void, GlobalMemoryDeleter>;

UniqueMemory allocGlobalMemory(size_t size, bool isStatic = false);

template <typename T>
class MemorySpan final {
private:
    std::shared_ptr<void> mMem;
    size_t mBegin, mEnd;
public:
    template <typename U>
    friend class MemorySpan;

    MemorySpan() : mBegin(0), mEnd(0) {}

    explicit MemorySpan(UniqueMemory memory, const size_t size)
        : mMem(std::move(memory)), mBegin(0), mEnd(size / sizeof(T)) {}

    explicit MemorySpan(const size_t size)
        : mMem(allocGlobalMemory(size * sizeof(T), true)), mBegin(0), mEnd(size) {}

    template <typename U>
    explicit MemorySpan(const MemorySpan<U>& rhs)
        : mMem(rhs.mMem), mBegin(rhs.mBegin * sizeof(U) / sizeof(T)),
        mEnd(rhs.mEnd * sizeof(U) / sizeof(T)) {
        #ifdef CAMERA_DEBUG
        if (rhs.mBegin % sizeof(T) != 0 || rhs.mEnd % sizeof(T) != 0)
            throw std::logic_error("bad cast");
        #endif
    }

    T* begin() const {
        return reinterpret_cast<T*>(mMem.get()) + mBegin;
    }

    T* end() const {
        return reinterpret_cast<T*>(mMem.get()) + mEnd;
    }

    size_t size() const {
        return mEnd - mBegin;
    }

    MemorySpan subSpan(const size_t begin, const size_t end = std::numeric_limits<size_t>::max()) const {
        #ifdef CAMERA_DEBUG
        if (mBegin + begin > mEnd)throw std::logic_error("bad cast");
        #endif
        MemorySpan res;
        res.mMem = mMem;
        res.mBegin = mBegin + begin;
        if (end == std::numeric_limits<size_t>::max())res.mEnd = mEnd;
        else res.mEnd = mBegin + end;
        return res;
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

    T operator*() const {
        return *get();
    }
};

template<typename T, size_t Rem = sizeof(T) % CACHE_ALIGN>
struct AlignedType final :T {
    unsigned char padding[CACHE_ALIGN - Rem];
};

template<typename T>
struct AlignedType<T, 0> final :T {};
