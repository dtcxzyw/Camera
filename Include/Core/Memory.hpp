#pragma once
#include <memory>
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <utility>
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

void gc();
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

    T* begin() const noexcept {
        return reinterpret_cast<T*>(mMem.get()) + mBegin;
    }

    T* data() const noexcept {
        return begin();
    }

    T* end() const noexcept {
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

    void memset(const int mask = 0) const {
        cudaMemset(begin(), mask, (mEnd - mBegin) * sizeof(T));
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
    size_t mSize;
public:
    explicit PinnedBuffer(const size_t size) : mMemory(size * sizeof(T)), mSize(size) {}

    T* begin() const noexcept {
        return reinterpret_cast<T*>(mMemory.get());
    }

    T* data() const noexcept {
        return begin();
    }

    T* end() const noexcept {
        return begin() + mSize;
    }

    T& operator[](const size_t idx) {
        return begin()[idx];
    }

    const T& operator*() const {
        return *begin();
    }

    T& operator*() {
        return *begin();
    }

    size_t size() const {
        return mSize;
    }
};

template <typename Container>
auto upload(const Container& data) {
    using T = typename std::decay<decltype(*std::data(data))>::type;
    MemorySpan<T> res(std::size(data));
    checkError(cudaMemcpy(res.begin(), std::data(data), std::size(data) * sizeof(T),
        cudaMemcpyHostToDevice));
    return res;
}
