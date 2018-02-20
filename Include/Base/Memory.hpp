#pragma once
#include <memory>
#include <Base/Common.hpp>

class Memory final : Uncopyable {
public:
    explicit Memory(size_t size);
    ~Memory();
    char* getPtr() const;
    size_t size() const;
private:
    void* mPtr;
    size_t mSize;
};

using UniqueMemory = std::unique_ptr<Memory>;
using SharedMemory = std::shared_ptr<Memory>;

template <typename T>
class DataViewer final {
private:
    SharedMemory mMem;
    T* mPtr;
    size_t mSize{};
public:
    DataViewer() = default;

    explicit DataViewer(const SharedMemory memory, const size_t offset = 0, const size_t size = 0)
        : mMem(memory), mPtr(reinterpret_cast<T*>(memory->getPtr() + offset)), mSize(size) {
        if (size == 0)
            mSize = (mMem->size() - offset) / sizeof(T);
    }

    T* begin() const {
        return mPtr;
    }

    T* end() const {
        return mPtr + mSize;
    }

    size_t size() const {
        return mSize;
    }
};

template <typename T>
auto allocBuffer(size_t size = 1) {
    return DataViewer<T>(std::make_shared<Memory>(size * sizeof(T)));
}

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
};
