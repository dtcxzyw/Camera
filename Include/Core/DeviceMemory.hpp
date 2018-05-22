#pragma once
#include <Core/Common.hpp>
#include <type_traits>
#include <Core/Memory.hpp>
#include <Core/Environment.hpp>

struct DeviceMemoryDesc final {
    void* ptr;
    unsigned int size;
    DEVICE DeviceMemoryDesc() :ptr(nullptr), size(0) {}
    template<typename T>
    DEVICE T* viewAs() {
        return static_cast<T*>(ptr);
    }
    template<typename T>
    DEVICE  const T* viewAs() const {
        return static_cast<const T*>(ptr);
    }
};

DEVICE DeviceMemoryDesc alloc(unsigned int size);
DEVICE void free(DeviceMemoryDesc desc);

/*
template <class T>
DEVICEINLINE T&& forward(typename std::remove_reference<T>::type& arg) noexcept {
    // forward an lvalue as either an lvalue or an rvalue
    return (static_cast<T&&>(arg));
}

template <class T>
DEVICEINLINE T&& forward(typename std::remove_reference<T>::type&& arg) noexcept {
    // forward an rvalue as an rvalue
    return (static_cast<T&&>(arg));
}

template <class T>
DEVICEINLINE typename std::remove_reference<T>::type&& move(T&& arg) noexcept {
    // forward arg as movable
    return (static_cast<typename std::remove_reference<T>::type&&>(arg));
}

template <typename T>
class UniquePtr final {
private:
    DeviceMemoryDesc mPtr;
public:
    DEVICE UniquePtr() = default;
    DEVICE explicit UniquePtr(const DeviceMemoryDesc& ptr) : mPtr(ptr) {}
    DEVICE UniquePtr(UniquePtr&& rhs) noexcept : mPtr(rhs.mPtr) {
        rhs.mPtr = nullptr;
    }

    DEVICE UniquePtr(const UniquePtr&) = delete;
    DEVICE UniquePtr& operator=(const UniquePtr&) = delete;
    DEVICE UniquePtr& operator=(UniquePtr&& rhs) noexcept {
        cudaSwap(mPtr, rhs.mPtr);
        return *this;
    }

    DEVICE T& operator*() {
        return *mPtr.viewAs<T>();
    }

    DEVICE const T& operator*() const {
        return *mPtr.viewAs<T>();
    }

    DEVICE const T* get() const {
        return mPtr.viewAs<T>();
    }

    DEVICE T* get() {
        return mPtr.viewAs<T>();
    }

    DEVICE const T* operator->() const {
        return get();
    }

    DEVICE T* operator->() {
        return get();
    }

    DEVICE ~UniquePtr() {
        free(mPtr);
    }
};

template <typename T, typename... Args>
DEVICEINLINE UniquePtr<T> makeUniquePtr(Args&&... args) {
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
    DEVICE SharedPtr() = default;
    DEVICE explicit SharedPtr(const DeviceMemoryDesc& ptr) : mPtr(ptr) {
        count() = 1U;
    }

    DEVICE SharedPtr(SharedPtr&& rhs) noexcept : mPtr(rhs.mPtr) {
        rhs.mPtr = nullptr;
    }

    DEVICE SharedPtr(const SharedPtr& rhs) : mPtr(rhs.mPtr) {
        if (this != &rhs)atomicInc(&count(), maxv);
    }

    DEVICE SharedPtr& operator=(const SharedPtr& rhs) {
        if (this != &rhs) {
            mPtr = rhs.mPtr;
            atomicInc(&count(), maxv);
        }
        return *this;
    }

    DEVICE SharedPtr& operator=(SharedPtr&& rhs) noexcept {
        cudaSwap(mPtr, rhs.mPtr);
        return *this;
    }

    DEVICE T& operator*() {
        return *mPtr.viewAs<T>();
    }

    DEVICE const T& operator*() const {
        return *mPtr.viewAs<T>();
    }

    DEVICE const T* get() const {
        return mPtr.viewAs<T>();
    }

    DEVICE T* get() {
        return mPtr.viewAs<T>();
    }

    DEVICE const T* operator->() const {
        return get();
    }

    DEVICE T* operator->() {
        return get();
    }

    DEVICE ~SharedPtr() {
        if (mPtr.ptr && atomicDec(&count(), maxv) == 1U) free(mPtr);
    }
};

template <typename T, typename... Args>
DEVICEINLINE SharedPtr<T> makeSharedPtr(Args&&... args) {
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
    DEVICE explicit VectorDevice(const unsigned int size = 0U)
        : mData(alloc(sizeof(T)*size)), mSize(size), mBufferSize(size) {}

    DEVICE VectorDevice(const VectorDevice&) = delete;
    DEVICE VectorDevice(VectorDevice&& rhs) noexcept
        : mData(rhs.mData), mSize(rhs.mSize), mBufferSize(rhs.mBufferSize) {
        rhs.mData = {};
    }
    DEVICE VectorDevice& operator=(const VectorDevice&) = delete;
    DEVICE VectorDevice& operator=(VectorDevice&& rhs) noexcept {
        free(mData);
        mData = rhs.mData;
        rhs.mData = {};
        mSize = rhs.mSize;
        mBufferSize = rhs.mBufferSize;
        return *this;
    }

    DEVICE const T& operator[](const unsigned int i) const {
        return mData.viewAs<T>()[i];
    }

    DEVICE T& operator[](const unsigned int i) {
        return mData.viewAs<T>()[i];
    }

    DEVICE unsigned int size() const {
        return mSize;
    }

    DEVICE void reserve(const unsigned int size) {
        const auto ptr = alloc(size * sizeof(T));
        memcpy(ptr.ptr, mData.ptr, sizeof(T)*mSize);
        free(mData);
        mData = ptr;
        mBufferSize = size;
    }

    template<typename... Args>
    DEVICE void emplaceBack(Args&&... args) {
        if (mSize == mBufferSize)reserve(mSize << 1);
        new(mData.viewAs<T>() + mSize) T(forward<Args>(args)...);
        ++mSize;
    }

    DEVICE T* begin() {
        return mData.viewAs<T>();
    }

    DEVICE T* end() {
        return mData.viewAs<T>() + mSize;
    }

    DEVICE const T* cbegin() const {
        return mData.viewAs<T>();
    }

    DEVICE const T* cend() const {
        return mData.viewAs<T>() + mSize;
    }

    DEVICE void clear() {
        free(mData);
        mData = {};
        mSize = mBufferSize = 0;
    }

    DEVICE ~VectorDevice() {
        free(mData);
    }
};
*/

template<typename T,typename... Args>
GLOBAL void constructKernel(T* ptr, Args... args) {
    new(ptr) T(args...);
}

template<typename T, typename... Args>
MemorySpan<T> constructOnDevice(Stream& stream, Args&&... args){
    MemorySpan<T> res(1);
    stream.launchDim(constructKernel<T, Args...>, {}, {}, res.begin(), std::forward<Args>(args)...);
    return res;
}
