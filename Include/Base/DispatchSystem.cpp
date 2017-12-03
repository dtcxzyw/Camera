#include "DispatchSystem.hpp"
#include "Constant.hpp"

namespace Impl {

    /*
    Any::Any(Any & rhs) {
        if (rhs.mData)mData = rhs.mData->clone();
    }

    bool Any::vaild() const {
        return mData.get() == nullptr;
    }

    void Any::release() {
        mData.reset();
    }

    void Any::swap(Any & rhs) {
        mData.swap(rhs.mData);
    }
    */

    size_t Impl::getPID() {
        thread_local static size_t cnt = 0;
        return ++cnt;
    }

    DeviceMemory::DeviceMemory(CommandBuffer& buffer, size_t size, MemoryType type)
        :mID(getID()), mSize(size), mBuffer(buffer), mType(type) {}

    DeviceMemory::~DeviceMemory() {
        mBuffer.newDMI(mID, mSize, getID(),mType);
    }

    size_t DeviceMemory::getID() const {
        return mID;
    }

    size_t DeviceMemory::size() const {
        return mSize;
    }

    DeviceMemoryInstance::DeviceMemoryInstance(size_t size, size_t end) :mSize(size), mEnd(end) {}

    bool DeviceMemoryInstance::shouldRelease(size_t current) const {
        return current > mEnd;
    }

    void DeviceMemoryInstance::set(int mask, Stream & stream) {}

    GlobalMemory::GlobalMemory(size_t size, size_t end):DeviceMemoryInstance(size,end) {}

    void * GlobalMemory::get() {
        if (!mMemory)mMemory = std::make_unique<Memory>(mSize);
        return mMemory->getPtr();
    }

    void GlobalMemory::set(const void * src, Stream & stream) {
        checkError(cudaMemcpyAsync(mMemory->getPtr(),src,mSize,
            cudaMemcpyDefault,stream.getID()));
    }

    void GlobalMemory::set(int mask, Stream & stream) {
        checkError(cudaMemsetAsync(mMemory->getPtr(),mask,mSize,stream.getID()));
    }

    ConstantMemory::ConstantMemory(size_t size, size_t end)
        :DeviceMemoryInstance(size,end),mPtr(nullptr) {}

    ConstantMemory::~ConstantMemory() {
        Impl::constantFree(mPtr,static_cast<unsigned int>(mSize));
    }

    void * ConstantMemory::get() {
        if (mPtr == nullptr)mPtr = Impl::constantAlloc(static_cast<unsigned int>(mSize));
        return mPtr;
    }

    void ConstantMemory::set(const void * src, Stream & stream) {
        Impl::constantSet(mPtr,src,static_cast<unsigned int>(mSize),stream.getID());
    }

}

void CommandBuffer::newDMI(size_t id, size_t size, size_t end, Impl::MemoryType type) {
    if (type == Impl::MemoryType::global)
        mDeviceMemory.emplace(id,std::make_unique<Impl::GlobalMemory>(size, end));
    else
        mDeviceMemory.emplace(id,std::make_unique<Impl::ConstantMemory>(size, end));
}
