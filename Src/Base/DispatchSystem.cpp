#include <Base/DispatchSystem.hpp>
#include <Base/Constant.hpp>
#include <algorithm>

namespace Impl {

    ID getPID() {
        thread_local static ID cnt = 0;
        return ++cnt;
    }

    DeviceMemory::DeviceMemory(CommandBuffer& buffer, size_t size, MemoryType type)
        :Resource(buffer), mSize(size), mType(type) {}

    DeviceMemory::~DeviceMemory() {
        if (mType == MemoryType::global)
            addInstance(std::make_unique<GlobalMemory>(mSize));
        else
            addInstance(std::make_unique<ConstantMemory>(mSize));
    }

    size_t DeviceMemory::size() const {
        return mSize;
    }

    void DeviceMemoryInstance::getRes(void * res,cudaStream_t) {
        *reinterpret_cast<void**>(res) = get();
    }

    DeviceMemoryInstance::DeviceMemoryInstance(size_t size):mSize(size) {}

    void DeviceMemoryInstance::memset(int mask, Stream & stream) {}

    GlobalMemory::GlobalMemory(size_t size):DeviceMemoryInstance(size) {}

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

    ConstantMemory::ConstantMemory(size_t size)
        :DeviceMemoryInstance(size),mPtr(nullptr) {}

    ConstantMemory::~ConstantMemory() {
        if(mPtr)Impl::constantFree(mPtr,static_cast<unsigned int>(mSize));
    }

    void * ConstantMemory::get() {
        if (mPtr == nullptr)mPtr = Impl::constantAlloc(static_cast<unsigned int>(mSize));
        return mPtr;
    }

    void ConstantMemory::set(const void * src, Stream & stream) {
        Impl::constantSet(get(),src,static_cast<unsigned int>(mSize),stream.getID());
    }

    DeviceMemoryInstance & Operator::getMemory(ID id) const {
        return dynamic_cast<DeviceMemoryInstance&>(mBuffer.getResource(id));
    }

    Operator::Operator(CommandBuffer & buffer):mBuffer(buffer),mID(getPID()) {}

    ID Operator::getID() {
        return mID;
    }

    Memset::Memset(CommandBuffer & buffer, ID memoryID, int mask)
        :Operator(buffer), mMemoryID(memoryID), mMask(mask) {}

    void Memset::emit(Stream & stream) {
        getMemory(mMemoryID).memset(mMask,stream);
    }

    Memcpy::Memcpy(CommandBuffer & buffer, ID dst
        , std::function<void(std::function<void(const void*)>)>&& src):
        Operator(buffer),mDst(dst),mSrc(src){}


    void Memcpy::emit(Stream& stream) {
        mSrc([this,&stream](auto ptr) {
            getMemory(mDst).set(ptr, stream);
        });
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

    void CastTag::get(CommandBuffer & buffer, ID id,void* ptr) {
        buffer.getResource(id).getRes(ptr,buffer.getStream());
    }

    DMRef::DMRef(const std::shared_ptr<Impl::DeviceMemory>& ref):ResourceRef(ref) {}
    size_t DMRef::size() const {
        return dynamic_cast<Impl::DeviceMemory&>(*mRef).size();
    }

    void CUDART_CB downloadCallback(cudaStream_t stream, cudaError_t status, void *userData) {
        auto info = reinterpret_cast<std::pair<unsigned int, std::atomic_uint*>*>(userData);
        *info->second = info->first;
        delete info;
    }

    void LaunchSize::download(std::atomic_uint & dst, CommandBuffer & buffer) {
        auto id = mHelper;
        auto tmp = new std::pair<unsigned int, std::atomic_uint*>(0,&dst);
        buffer.pushOperator([id,&buffer,tmp](Stream& stream) {
            checkError(cudaMemcpyAsync(&tmp->first,Impl::cast(id,buffer),sizeof(unsigned int),
                cudaMemcpyDeviceToHost,stream.getID()));
        });
        buffer.addCallback(downloadCallback,tmp);
    }
}

void CommandBuffer::registerResource(ID id, std::unique_ptr<ResourceInstance>&& instance) {
    mResource.emplace(id, std::move(instance));
}

void CommandBuffer::setPromise(const std::shared_ptr<bool>& promise) {
    mPromise = promise;
    addCallback(Impl::streamCallback, promise.get());
    mLast = 0;
}

void CommandBuffer::memset(Impl::DMRef& memory, int mark) {
    mCommandQueue.emplace(std::make_unique<Impl::Memset>(*this, memory.getID(),mark));
}

void CommandBuffer::memcpy(Impl::DMRef & dst
    , std::function<void(std::function<void(const void*)>)>&& src) {
    mCommandQueue.emplace(std::make_unique<Impl::Memcpy>(*this, dst.getID(), std::move(src)));
}

void CommandBuffer::addCallback(cudaStreamCallback_t func, void * data) {
    pushOperator([=](Stream& stream) {
        checkError(cudaStreamAddCallback(stream.getID(), func, data, 0));
    });
}

void CommandBuffer::pushOperator(std::unique_ptr<Impl::Operator>&& op) {
    mCommandQueue.emplace(std::move(op));
}

void CommandBuffer::pushOperator(std::function<void(Stream&)>&& op) {
    mCommandQueue.emplace(std::make_unique<FunctionOperator>(*this,std::move(op)));
}

void CommandBuffer::update(Stream& stream) {
    if (!mPromise)throw std::exception("The command buffer is imcomplete.");
    if (!mCommandQueue.empty()) {
        mStream = stream.getID();
        mCommandQueue.front()->emit(stream);
        std::vector<ID> list;
        for (auto&& x : mResource)
            if (x.second->shouldRelease(mLast))
                list.emplace_back(x.first);
        for (auto&& id : list)
            mResource.erase(id);
        mLast = mCommandQueue.front()->getID();
        mCommandQueue.pop();
    }
}

bool CommandBuffer::finished() const {
    return mCommandQueue.empty();
}

bool CommandBuffer::isDone() const {
    return *mPromise;
}

ResourceInstance & CommandBuffer::getResource(ID id) {
    return *mResource.find(id)->second;
}

cudaStream_t CommandBuffer::getStream() {
    return mStream;
}

void DispatchSystem::update(Clock::time_point t) {
    auto& stream = *std::min_element(mStreams.begin(), mStreams.end());
    if (stream.free() && !mTasks.empty()) {
        stream.set(std::move(mTasks.front()));
        mTasks.pop();
    }
    stream.update(t);
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
        update(t);
        if (t - begin > tot)return;
        if (mTasks.empty() && std::all_of(mStreams.cbegin(), mStreams.cend(), [](const auto& info) {
            return info.free();
        })) return;
    }
}

void DispatchSystem::update() {
    update(Clock::now());
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

bool DispatchSystem::StreamInfo::free() const {
    return mTask==nullptr && mPool.empty();
}

void DispatchSystem::StreamInfo::set(std::unique_ptr<CommandBuffer>&& task) {
    mTask = std::move(task);
}

void DispatchSystem::StreamInfo::update(Clock::time_point point) {
    mPool.erase(std::remove_if(mPool.begin(), mPool.end(),
        [](auto&& task) {return task->isDone(); }), mPool.end());
    if (mTask) {
        mLast = point;
        mTask->update(mStream);
        if (mTask->finished()) {
            mPool.emplace_back();
            mPool.back().swap(mTask);
        }
    }
}

bool DispatchSystem::StreamInfo::operator<(const StreamInfo & rhs) const {
    return mLast<rhs.mLast;
}

ResourceInstance::ResourceInstance():mEnd(Impl::getPID()) {}

bool ResourceInstance::shouldRelease(ID current) const {
    return current>mEnd;
}
