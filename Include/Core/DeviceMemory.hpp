#pragma once
#include <Core/Common.hpp>
#include <type_traits>
#include <Core/Memory.hpp>
#include <Core/Environment.hpp>

struct DeviceMemoryDesc final {
    void* ptr;
    unsigned int size;
    CUDA DeviceMemoryDesc() :ptr(nullptr), size(0) {}
    template<typename T>
    CUDA T* viewAs() {
        return static_cast<T*>(ptr);
    }
    template<typename T>
    CUDA  const T* viewAs() const {
        return static_cast<const T*>(ptr);
    }
};

CUDA DeviceMemoryDesc alloc(unsigned int size);
CUDA void free(DeviceMemoryDesc desc);

template <class T>
CUDAINLINE T&& forward(typename std::remove_reference<T>::type& arg) noexcept {
    // forward an lvalue as either an lvalue or an rvalue
    return (static_cast<T&&>(arg));
}

template <class T>
CUDAINLINE T&& forward(typename std::remove_reference<T>::type&& arg) noexcept {
    // forward an rvalue as an rvalue
    return (static_cast<T&&>(arg));
}

template <class T>
CUDAINLINE typename std::remove_reference<T>::type&& move(T&& arg) noexcept {
    // forward arg as movable
    return (static_cast<typename std::remove_reference<T>::type&&>(arg));
}

template <typename T>
class UniquePtr final {
private:
    DeviceMemoryDesc mPtr;
public:
    CUDA UniquePtr() = default;
    CUDA explicit UniquePtr(const DeviceMemoryDesc& ptr) : mPtr(ptr) {}
    CUDA UniquePtr(UniquePtr&& rhs) noexcept : mPtr(rhs.mPtr) {
        rhs.mPtr = nullptr;
    }

    CUDA UniquePtr(const UniquePtr&) = delete;
    CUDA UniquePtr& operator=(const UniquePtr&) = delete;
    CUDA UniquePtr& operator=(UniquePtr&& rhs) noexcept {
        mPtr = rhs.mPtr;
        rhs.mPtr = nullptr;
        return *this;
    }

    CUDA T& operator*() {
        return *mPtr.viewAs<T>();
    }

    CUDA const T& operator*() const {
        return *mPtr.viewAs<T>();
    }

    CUDA const T* get() const {
        return mPtr.viewAs<T>();
    }

    CUDA T* get() {
        return mPtr.viewAs<T>();
    }

    CUDA const T* operator->() const {
        return get();
    }

    CUDA T* operator->() {
        return get();
    }

    CUDA ~UniquePtr() {
        free(mPtr);
    }
};

template <typename T, typename... Args>
CUDAINLINE UniquePtr<T> makeUniquePtr(Args&&... args) {
    auto desc = alloc(sizeof(T));
    new(desc.ptr) T(forward<Args>(args)...);
    return UniquePtr<T>(desc);
}

template<typename T>
class SharedPtr final {
private:
    DeviceMemoryDesc mPtr;
    unsigned int& count() {
         return mPtr.viewAs<std::pair<T, unsigned int>>()->second;
    }
public:
    CUDA SharedPtr() = default;
    CUDA explicit SharedPtr(const DeviceMemoryDesc& ptr) : mPtr(ptr) {
        count() = 1U;
    }

    CUDA SharedPtr(SharedPtr&& rhs) noexcept : mPtr(rhs.mPtr) {
        rhs.mPtr = nullptr;
    }

    CUDA SharedPtr(const SharedPtr& rhs) : mPtr(rhs.mPtr) {
        atomicInc(&count(), maxv);
    }

    CUDA SharedPtr& operator=(const SharedPtr& rhs) {
        mPtr = rhs.mPtr;
        atomicInc(&count(), maxv);
        return *this;
    }

    CUDA SharedPtr& operator=(SharedPtr&& rhs) noexcept {
        mPtr = rhs.mPtr;
        rhs.mPtr = nullptr;
        return *this;
    }

    CUDA T& operator*() {
        return *mPtr.viewAs<T>();
    }

    CUDA const T& operator*() const {
        return *mPtr.viewAs<T>();
    }

    CUDA const T* get() const {
        return mPtr.viewAs<T>();
    }

    CUDA T* get() {
        return mPtr.viewAs<T>();
    }

    CUDA const T* operator->() const {
        return get();
    }

    CUDA T* operator->() {
        return get();
    }

    CUDA ~SharedPtr() {
        if (mPtr.ptr && atomicDec(&count(), maxv) == 1U) free(mPtr);
    }
};

template <typename T, typename... Args>
CUDAINLINE SharedPtr<T> makeSharedPtr(Args&&... args) {
    auto desc = alloc(sizeof(unsigned int) + sizeof(T));
    new(desc.ptr) T(forward<Args>(args)...);
    return SharedPtr<T>(desc);
}

template <typename T>
class VectorDevice final {
private:
    DeviceMemoryDesc mData;
    unsigned int mSize, mBufferSize;
public:
    CUDA explicit VectorDevice(const unsigned int size = 0U)
        : mData(alloc(sizeof(T)*size)), mSize(size), mBufferSize(size) {}

    CUDA VectorDevice(const VectorDevice&) = delete;
    CUDA VectorDevice(VectorDevice&& rhs) noexcept
        : mData(rhs.mData), mSize(rhs.mSize), mBufferSize(rhs.mBufferSize) {
        rhs.mData = {};
    }
    CUDA VectorDevice& operator=(const VectorDevice&) = delete;
    CUDA VectorDevice& operator=(VectorDevice&& rhs) noexcept {
        free(mData);
        mData = rhs.mData;
        rhs.mData = {};
        mSize = rhs.mSize;
        mBufferSize = rhs.mBufferSize;
        return *this;
    }

    CUDA const T& operator[](const unsigned int i) const {
        return mData.viewAs<T>()[i];
    }

    CUDA T& operator[](const unsigned int i) {
        return mData.viewAs<T>()[i];
    }

    CUDA unsigned int size() const {
        return mSize;
    }

    CUDA void reserve(const unsigned int size) {
        const auto ptr = alloc(size * sizeof(T));
        memcpy(ptr.ptr, mData.ptr, sizeof(T)*mSize);
        free(mData);
        mData = ptr;
        mBufferSize = size;
    }

    template<typename... Args>
    CUDA void emplaceBack(Args&&... args) {
        if (mSize == mBufferSize)reserve(mSize << 1);
        new(mData.viewAs<T>() + mSize) T(forward<Args>(args)...);
        ++mSize;
    }

    CUDA T* begin() {
        return mData.viewAs<T>();
    }

    CUDA T* end() {
        return mData.viewAs<T>() + mSize;
    }

    CUDA const T* cbegin() const {
        return mData.viewAs<T>();
    }

    CUDA const T* cend() const {
        return mData.viewAs<T>() + mSize;
    }

    CUDA void clear() {
        free(mData);
        mData = {};
        mSize = mBufferSize = 0;
    }

    CUDA ~VectorDevice() {
        free(mData);
    }
};

template<typename T,typename... Args>
GLOBAL void constructKernel(T* ptr, Args... args) {
    new(ptr) T(args...);
}

template<typename T, typename... Args>
MemorySpan<T> constructOnDevice(CommandBuffer& buffer, Args&&... args){
    MemorySpan<T> res(1);
    buffer.callKernel(constructKernel<T,Args...>, buffer.useAllocated(res), std::forward<Args>(args)...);
    return res;
}
