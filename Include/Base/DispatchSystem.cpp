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

    void DeviceMemoryInstance::memset(int mask, Stream & stream) {}

    GlobalMemory::GlobalMemory(size_t size, ID end):DeviceMemoryInstance(size,end) {}

    void * GlobalMemory::get() {
        if (!mMemory)mMemory = std::make_unique<Memory>(mSize);
        return mMemory->getPtr();
    }

    void GlobalMemory::set(const void * src, Stream & stream) {
        checkError(cudaMemcpyAsync(get(),src,mSize,cudaMemcpyDefault,stream.getID()));
    }

    void GlobalMemory::memset(int mask, Stream & stream) {
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
        getMemory(mMemoryID).memset(mMask,stream);
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
    void* CastTag::get(CommandBuffer & buffer, ID id) {
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

bool CommandBuffer::finished() const {
    return mCommandQueue.empty();
}

DispatchSystem::DispatchSystem(size_t size):mStreams(size){}

Future DispatchSystem::submit(std::unique_ptr<CommandBuffer>&& buffer) {
    mTasks.push(std::move(buffer));
    auto promise = std::make_shared<bool>(false);
    mTasks.back()->setPromise(promise);
    return promise;
}

size_t DispatchSystem::size() const {
    return mTasks.size();
}

void DispatchSystem::update(std::chrono::nanoseconds tot) {
    auto begin = Clock::now();
    while (true) {
        auto t = Clock::now();
        auto& stream = *std::min_element(mStreams.begin(),mStreams.end());
        if (stream.free() && !mTasks.empty()) {
            stream.set(std::move(mTasks.front()));
            mTasks.pop();
        }
        if (t - begin > tot)return;
        stream.update(t);
    }
}

Future::Future(const std::shared_ptr<bool>& promise):mPromise(promise) {}

bool Future::finished() const {
    return *mPromise;
}

FunctionOperator::FunctionOperator(CommandBuffer& buffer,
    std::function<void(Stream&)>&& closure):Operator(buffer),mClosure(closure) {}

void FunctionOperator::emit(Stream & stream) {
    mClosure(stream);
}

DispatchSystem::StreamInfo::StreamInfo():mLast(Clock::now()) {}

bool DispatchSystem::StreamInfo::free() {
    return mTask==nullptr;
}

void DispatchSystem::StreamInfo::set(std::unique_ptr<CommandBuffer>&& task) {
    mTask = std::move(task);
}

void DispatchSystem::StreamInfo::update(Clock::time_point point) {
    if (mTask) {
        mLast = point;
        mTask->update(mStream);
        if (mTask->finished()) {
            mPool.emplace_back();
            mPool.back().swap(mTask);
        }
    }
    mPool.erase(std::remove_if(mPool.begin(), mPool.end(), 
        [](auto&& task) {return *task->mPromise; }), mPool.end());
}

bool DispatchSystem::StreamInfo::operator<(const StreamInfo & rhs) const {
    return mLast<rhs.mLast;
}
