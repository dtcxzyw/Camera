#include "DispatchSystem.hpp"
#include "Constant.hpp"
#include <algorithm>

namespace Impl {

    ID getPID() {
        thread_local static ID cnt = 0;
        return ++cnt;
    }

    DeviceMemory::DeviceMemory(CommandBuffer& buffer, size_t size, MemoryType type)
        :mID(getID()), mSize(size), mBuffer(buffer), mType(type) {}

    DeviceMemory::~DeviceMemory() {
        mBuffer.newDMI(mID, mSize, getID(),mType);
    }

    ID DeviceMemory::getID() const {
        return mID;
    }

    size_t DeviceMemory::size() const {
        return mSize;
    }

    DeviceMemoryInstance::DeviceMemoryInstance(size_t size, ID end) :mSize(size), mEnd(end) {}

    bool DeviceMemoryInstance::shouldRelease(ID current) const {
        return current > mEnd;
    }

    void DeviceMemoryInstance::set(int mask, Stream & stream) {}

    GlobalMemory::GlobalMemory(size_t size, ID end):DeviceMemoryInstance(size,end) {}

    void * GlobalMemory::get() {
        if (!mMemory)mMemory = std::make_unique<Memory>(mSize);
        return mMemory->getPtr();
    }

    void GlobalMemory::set(const void * src, Stream & stream) {
        checkError(cudaMemcpyAsync(get(),src,mSize,cudaMemcpyDefault,stream.getID()));
    }

    void GlobalMemory::set(int mask, Stream & stream) {
        checkError(cudaMemsetAsync(get(),mask,mSize,stream.getID()));
    }

    ConstantMemory::ConstantMemory(size_t size, ID end)
        :DeviceMemoryInstance(size,end),mPtr(nullptr) {}

    ConstantMemory::~ConstantMemory() {
        Impl::constantFree(mPtr,static_cast<unsigned int>(mSize));
    }

    void * ConstantMemory::get() {
        if (mPtr == nullptr)mPtr = Impl::constantAlloc(static_cast<unsigned int>(mSize));
        return mPtr;
    }

    void ConstantMemory::set(const void * src, Stream & stream) {
        Impl::constantSet(get(),src,static_cast<unsigned int>(mSize),stream.getID());
    }

    Operator::Operator(CommandBuffer & buffer):mBuffer(buffer),mID(getID()) {}

    ID Operator::getID() {
        return mID;
    }

    Impl::DeviceMemoryInstance & Operator::getMemory(ID memoryID) {
        return *mBuffer.mDeviceMemory.find(memoryID)->second;
    }

    Memset::Memset(CommandBuffer & buffer, ID memoryID, int mask)
        :Operator(buffer), mMemoryID(memoryID), mMask(mask) {}

    void Memset::emit(Stream & stream) {
        getMemory(mMemoryID).set(mMask,stream);
    }

    Memcpy::Memcpy(CommandBuffer & buffer, ID dst, std::function<void*()>&& src):
        Operator(buffer),mDst(dst),mSrc(src){}

    void Memcpy::emit(Stream & stream) {
        getMemory(mDst).set(mSrc(),stream);
    }

    void KernelLaunchDim::emit(Stream & stream) {
        mClosure(stream);
    }

    void KernelLaunchLinear::emit(Stream& stream) {
        mClosure(stream);
    }
    void CUDART_CB streamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
        *reinterpret_cast<bool*>(userData) = true;
    }
    void* IDTag::get(CommandBuffer & buffer, ID id) {
        return buffer.mDeviceMemory.find(id)->second->get();
    }
}

void CommandBuffer::newDMI(ID id, size_t size, ID end, Impl::MemoryType type) {
    if (type == Impl::MemoryType::global)
        mDeviceMemory.emplace(id,std::make_unique<Impl::GlobalMemory>(size, end));
    else
        mDeviceMemory.emplace(id,std::make_unique<Impl::ConstantMemory>(size, end));
}

void CommandBuffer::setPromise(const std::shared_ptr<bool>& promise) {
    mPromise = promise;
    pushOperator([=](Stream& stream) {
        checkError(cudaStreamAddCallback(stream.getID(), Impl::streamCallback, promise.get(), 0));
    });
}

CommandBuffer::CommandBuffer():mLast(0) {}

void CommandBuffer::memset(Impl::DMRef& memory, int mark) {
    mCommandQueue.emplace(std::make_unique<Impl::Memset>(*this, memory.getID(),mark));
}

void CommandBuffer::memcpy(Impl::DMRef & dst, std::function<void*()>&& src) {
    mCommandQueue.emplace(std::make_unique<Impl::Memcpy>(*this, dst.getID(), std::move(src)));
}

void CommandBuffer::pushOperator(std::unique_ptr<Impl::Operator>&& op) {
    mCommandQueue.emplace(std::move(op));
}

void CommandBuffer::pushOperator(std::function<void(Stream&)>&& op) {
    mCommandQueue.emplace(std::make_unique<FunctionOperator>(*this,std::move(op)));
}

void CommandBuffer::update(Stream& stream) {
    if (!mPromise)throw std::exception("The task is imcomplete.");
    if (!mCommandQueue.empty()) {
        mCommandQueue.front()->emit(stream);
        std::vector<ID> list;
        for (auto&& x : mDeviceMemory)
            if (x.second->shouldRelease(mLast))
                list.emplace_back(x.first);
        for (auto&& id : list)
            mDeviceMemory.erase(id);
        mLast = mCommandQueue.front()->getID();
        mCommandQueue.pop();
    }
}

bool CommandBuffer::ready() const {
    return *mPromise;
}

DispatchSystem::DispatchSystem(size_t size):mStreams(size),mPos(0),mAlloc(0) {}

Future DispatchSystem::submit(std::unique_ptr<CommandBuffer>&& buffer) {
    mTasks.emplace_back(mAlloc,std::move(buffer));
    mAlloc = (mAlloc + 1) % mStreams.size();
    auto promise = std::make_shared<bool>(false);
    mTasks.back().second->setPromise(promise);
    return promise;
}

size_t DispatchSystem::size() const {
    return mTasks.size();
}

void DispatchSystem::update(std::chrono::nanoseconds tot) {
    using Clock = std::chrono::high_resolution_clock;
    mPos %= mTasks.size();
    auto begin = Clock::now();
    for (;Clock::now()-begin>tot; mPos=(mPos+1)%mTasks.size())
        mTasks[mPos].second->update(mStreams[mTasks[mPos].first]);
    mTasks.resize(std::remove_if(mTasks.begin(), mTasks.end(), 
        [](auto&& x) {return x.second->ready(); }) - mTasks.begin());
}

Future::Future(const std::shared_ptr<bool>& promise):mPromise(promise) {}

bool Future::ready() const {
    return *mPromise;
}

FunctionOperator::FunctionOperator(CommandBuffer& buffer,
    std::function<void(Stream&)>&& closure):Operator(buffer),mClosure(closure) {}

void FunctionOperator::emit(Stream & stream) {
    mClosure(stream);
}
