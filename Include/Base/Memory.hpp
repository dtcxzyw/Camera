#pragma once
#include <memory>
#include <Base/Common.hpp>

class Memory final :Uncopyable {
public:
    Memory(size_t size);
    ~Memory();
    char* getPtr() const;
    size_t size() const;
private:
    void* mPtr;
    size_t mSize;
};

using UniqueMemory = std::unique_ptr<Memory>;
using SharedMemory = std::shared_ptr<Memory>;

template<typename T>
class DataViewer final {
private:
    SharedMemory mMem;
    T* mPtr;
    size_t mSize;
public:
    DataViewer() = default;
    DataViewer(SharedMemory memory, size_t offset = 0, size_t size = 0)
        :mMem(memory), mPtr(reinterpret_cast<T*>(memory->getPtr() + offset)), mSize(size) {
        if (size == 0)
            mSize = (mMem->size() - offset) / sizeof(T);
    }
    T* begin() const {
        return mPtr;
    }
    T* end() const {
        return mPtr + mSize;
    }
    decltype(auto) operator[](size_t i) {
        return *(mPtr + i);
    }
    decltype(auto) operator->() {
        return mPtr;
    }
    decltype(auto) operator*() {
        return *mPtr;
    }
    size_t size() const {
        return mSize;
    }
};

template<typename T>
auto allocBuffer(size_t size = 1) {
    return DataViewer<T>(std::make_shared<Memory>(size * sizeof(T)));
}
