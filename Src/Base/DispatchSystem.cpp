#include <Base/DispatchSystem.hpp>
#include <Base/Constant.hpp>
#include <algorithm>
#include <utility>

namespace Impl {

    ID getPID() {
        thread_local static ID cnt = 0;
        return ++cnt;
    }

    DeviceMemory::DeviceMemory(CommandBuffer& buffer, const size_t size, const MemoryType type)
        :Resource(buffer), mSize(size), mType(type) {}

    DeviceMemory::~DeviceMemory() {
        if (mType == MemoryType::Global)
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

    DeviceMemoryInstance::DeviceMemoryInstance(const size_t size):mSize(size) {}

    void DeviceMemoryInstance::memset(int, Stream &) {}

    GlobalMemory::GlobalMemory(const size_t size):DeviceMemoryInstance(size) {}

    void * GlobalMemory::get() {
        if (!mMemory)mMemory = std::make_unique<Memory>(mSize);
        return mMemory->getPtr();
    }

    void GlobalMemory::set(const void * src, Stream & stream) {
        checkError(cudaMemcpyAsync(get(),src,mSize,cudaMemcpyDefault,stream.get()));
    }

    void GlobalMemory::memset(const int mask, Stream & stream) {
        checkError(cudaMemsetAsync(get(),mask,mSize,stream.get()));
    }

    ConstantMemory::ConstantMemory(const size_t size)
        :DeviceMemoryInstance(size),mPtr(nullptr) {}

    ConstantMemory::~ConstantMemory() {
        if(mPtr)constantFree(mPtr,static_cast<unsigned int>(mSize));
    }

    void * ConstantMemory::get() {
        if (mPtr == nullptr)mPtr = constantAlloc(static_cast<unsigned int>(mSize));
        return mPtr;
    }

    void ConstantMemory::set(const void * src, Stream & stream) {
        constantSet(get(),src,static_cast<unsigned int>(mSize),stream.get());
    }

    DeviceMemoryInstance & Operator::getMemory(const ID id) const {
        return dynamic_cast<DeviceMemoryInstance&>(mBuffer.getResource(id));
    }

    Operator::Operator(CommandBuffer & buffer):mBuffer(buffer),mID(getPID()) {}

    ID Operator::getID() const {
        return mID;
    }

    Memset::Memset(CommandBuffer & buffer, const ID memoryID, const int mask)
        :Operator(buffer), mMemoryID(memoryID), mMask(mask) {}

    void Memset::emit(Stream & stream) {
        getMemory(mMemoryID).memset(mMask,stream);
    }

    Memcpy::Memcpy(CommandBuffer & buffer, const ID dst
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

    static void CUDART_CB streamCallback(cudaStream_t, cudaError_t, void *userData) {
        *reinterpret_cast<bool*>(userData) = true;
    }

    void CastTag::get(CommandBuffer & buffer, const ID id,void* ptr) {
        buffer.getResource(id).getRes(ptr,buffer.getStream());
    }

    DMRef::DMRef(const std::shared_ptr<DeviceMemory>& ref):ResourceRef(ref) {}
    size_t DMRef::size() const {
        return dynamic_cast<DeviceMemory&>(*mRef).size();
    }

    static void CUDART_CB downloadCallback(cudaStream_t, cudaError_t, void *userData) {
        const auto info = reinterpret_cast<std::pair<unsigned int, std::atomic_uint*>*>(userData);
        *info->second = info->first;
        delete info;
    }

    void LaunchSize::download(std::atomic_uint & dst, CommandBuffer & buffer) const {
        auto id = mHelper;
        auto tmp = new std::pair<unsigned int, std::atomic_uint*>(0,&dst);
        buffer.pushOperator([id,&buffer,tmp](Stream& stream) {
            checkError(cudaMemcpyAsync(&tmp->first,cast(id,buffer),sizeof(unsigned int),
                cudaMemcpyDeviceToHost,stream.get()));
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
        checkError(cudaStreamAddCallback(stream.get(), func, data, 0));
    });
}

void CommandBuffer::sync() {
    pushOperator([](Stream& stream) {stream.sync(); });
}

void CommandBuffer::pushOperator(std::unique_ptr<Impl::Operator>&& op) {
    mCommandQueue.emplace(std::move(op));
}

void CommandBuffer::pushOperator(std::function<void(Stream&)>&& op) {
    mCommandQueue.emplace(std::make_unique<FunctionOperator>(*this,std::move(op)));
}

namespace Impl {
    void CUDART_CB updateLast(cudaStream_t, cudaError_t, void *userData) {
        const auto info=reinterpret_cast<std::pair<CommandBuffer*, ID>*>(userData);
        info->first->mLast = info->second;
        delete info;
    }
}

void CommandBuffer::update(Stream& stream) {
    if (!mCommandQueue.empty()) {
        mStream = stream.get();
        auto&& command = mCommandQueue.front();
        command->emit(stream);
        if ((mCommandQueue.size() & 0b1111) ==0) {
            const auto ptr = new std::pair<CommandBuffer*, ID>(this, command->getID());
            checkError(cudaStreamAddCallback(mStream, Impl::updateLast, ptr, 0));
        }
        mCommandQueue.pop();
        if (mLast != mUpdated) {
            std::vector<ID> list;
            for (auto&& x : mResource)
                if (x.second->shouldRelease(mLast))
                    list.emplace_back(x.first);
            for (auto&& id : list)
                mResource.erase(id);
            mLast = mUpdated;
        }
    }
}

bool CommandBuffer::finished() const {
    return mCommandQueue.empty();
}

bool CommandBuffer::isDone() const {
    return *mPromise;
}

ResourceInstance & CommandBuffer::getResource(const ID id) {
    return *mResource.find(id)->second;
}

cudaStream_t CommandBuffer::getStream() const {
    return mStream;
}

CommandBuffer::CommandBuffer(): mLast(0), mUpdated(0) {}

void DispatchSystem::update(const Clock::time_point t) {
    auto& stream = *std::min_element(mStreams.begin(), mStreams.end());
    if (stream.free() && !mTasks.empty()) {
        stream.set(std::move(mTasks.front()));
        mTasks.pop();
    }
    stream.update(t);
}

DispatchSystem::DispatchSystem(const size_t size):mStreams(size){}

Future DispatchSystem::submit(std::unique_ptr<CommandBuffer>&& buffer) {
    mTasks.push(std::move(buffer));
    const auto promise = std::make_shared<bool>(false);
    mTasks.back()->setPromise(promise);
    return Future{ promise };
}

size_t DispatchSystem::size() const {
    return mTasks.size();
}

void DispatchSystem::update(const std::chrono::nanoseconds tot) {
    const auto end = Clock::now()+tot;
    while (true) {
        const auto t = Clock::now();
        if (t > end)return;
        update(t);
        if (mTasks.empty() && std::all_of(mStreams.cbegin(), mStreams.cend(), [](const auto& info) {
            return info.free();
        })) return;
    }
}

void DispatchSystem::update() {
    update(Clock::now());
}

Future::Future(std::shared_ptr<bool> promise):mPromise(std::move(promise)) {}

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

void DispatchSystem::StreamInfo::update(const Clock::time_point point) {
    mPool.erase(std::remove_if(mPool.begin(), mPool.end(),
        [](auto&& task) {return task->isDone(); }), mPool.end());
    if (mTask) {
        mLast = point;
        mTask->update(mStream);
        if (mTask->finished())
            mPool.emplace_back(std::move(mTask));
    }
}

bool DispatchSystem::StreamInfo::operator<(const StreamInfo & rhs) const {
    return mLast<rhs.mLast;
}

ResourceInstance::ResourceInstance():mEnd(Impl::getPID()) {}

bool ResourceInstance::shouldRelease(const ID current) const {
    return current>mEnd;
}
