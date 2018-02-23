#include <Base/DispatchSystem.hpp>
#include <Base/Constant.hpp>
#include <algorithm>
#include <utility>

namespace Impl {

    ID getPID() {
        thread_local static ID cnt = 0;
        return ++cnt;
    }

    DeviceMemory::DeviceMemory(ResourceManager& manager, const size_t size, const MemoryType type)
        : Resource(manager), mSize(size), mType(type) {}

    DeviceMemory::~DeviceMemory() {
        if (mType == MemoryType::Global)
            addInstance(std::make_unique<GlobalMemory>(mSize));
        else
            addInstance(std::make_unique<ConstantMemory>(mSize));
    }

    size_t DeviceMemory::size() const {
        return mSize;
    }

    void DeviceMemoryInstance::getRes(void* res, cudaStream_t) {
        *reinterpret_cast<void**>(res) = get();
    }

    DeviceMemoryInstance::DeviceMemoryInstance(const size_t size): mSize(size) {}

    void DeviceMemoryInstance::memset(int, Stream&) {}

    GlobalMemory::GlobalMemory(const size_t size): DeviceMemoryInstance(size) {}

    void* GlobalMemory::get() {
        if (!mMemory)mMemory = std::make_unique<Memory>(mSize);
        return mMemory->getPtr();
    }

    void GlobalMemory::set(const void* src, Stream& stream) {
        checkError(cudaMemcpyAsync(get(), src, mSize, cudaMemcpyDefault, stream.get()));
    }

    void GlobalMemory::memset(const int mask, Stream& stream) {
        checkError(cudaMemsetAsync(get(), mask, mSize, stream.get()));
    }

    ConstantMemory::ConstantMemory(const size_t size)
        : DeviceMemoryInstance(size), mPtr(nullptr) {}

    ConstantMemory::~ConstantMemory() {
        if (mPtr)constantFree(mPtr, static_cast<unsigned int>(mSize));
    }

    void* ConstantMemory::get() {
        if (mPtr == nullptr)mPtr = constantAlloc(static_cast<unsigned int>(mSize));
        return mPtr;
    }

    void ConstantMemory::set(const void* src, Stream& stream) {
        constantSet(get(), src, static_cast<unsigned int>(mSize), stream.get());
    }

    DeviceMemoryInstance& Operator::getMemory(const ID id) const {
        return dynamic_cast<DeviceMemoryInstance&>(mManager.getResource(id));
    }

    Operator::Operator(ResourceManager& manager): mManager(manager), mID(getPID()) {}

    ID Operator::getID() const {
        return mID;
    }

    Memset::Memset(ResourceManager& manager, const ID memoryID, const int mask)
        : Operator(manager), mMemoryID(memoryID), mMask(mask) {}

    void Memset::emit(Stream& stream) {
        getMemory(mMemoryID).memset(mMask, stream);
    }

    Memcpy::Memcpy(ResourceManager& manager, const ID dst
                   , std::function<void(std::function<void(const void*)>)>&& src)
        : Operator(manager), mDst(dst), mSrc(src) {}

    void Memcpy::emit(Stream& stream) {
        mSrc([this,&stream](auto ptr) {
            getMemory(mDst).set(ptr, stream);
        });
    }

    void KernelLaunchDim::emit(Stream& stream) {
        mClosure(stream);
    }

    void KernelLaunchLinear::emit(Stream& stream) {
        mClosure(stream);
    }

    void CastTag::get(ResourceManager& manager, const ID id, void* ptr) {
        manager.getResource(id).getRes(ptr, manager.getStream());
    }

    DMRef::DMRef(const std::shared_ptr<DeviceMemory>& ref): ResourceRef(ref) {}

    size_t DMRef::size() const {
        return dynamic_cast<DeviceMemory&>(*mRef).size();
    }

    static void CUDART_CB downloadCallback(cudaStream_t, cudaError_t, void* userData) {
        const auto info = reinterpret_cast<std::pair<unsigned int, std::atomic_uint*>*>(userData);
        *info->second = info->first;
        delete info;
    }

    void LaunchSize::download(std::atomic_uint& dst, CommandBuffer& buffer) const {
        auto id = mHelper;
        auto tmp = new std::pair<unsigned int, std::atomic_uint*>(0, &dst);
        auto&& manager = buffer.getResourceManager();
        buffer.pushOperator([id,&manager,tmp](Stream& stream) {
            checkError(cudaMemcpyAsync(&tmp->first, cast(id, manager),
                                       sizeof(unsigned int), cudaMemcpyDeviceToHost, stream.get()));
        });
        buffer.addCallback(downloadCallback, tmp);
    }
}

void ResourceManager::registerResource(ID id, std::unique_ptr<ResourceInstance>&& instance) {
    mResources.emplace(id, std::move(instance));
}

void CommandBuffer::memset(Impl::DMRef& memory, int mark) {
    mCommandQueue.emplace(std::make_unique<Impl::Memset>(*mResourceManager,
        memory.getID(), mark));
}

void CommandBuffer::memcpy(Impl::DMRef& dst
                           , std::function<void(std::function<void(const void*)>)>&& src) {
    mCommandQueue.emplace(std::make_unique<Impl::Memcpy>(*mResourceManager,
        dst.getID(),std::move(src)));
}

void CommandBuffer::addCallback(cudaStreamCallback_t func, void* data) {
    pushOperator([=](Stream& stream) {
        checkError(cudaStreamAddCallback(stream.get(), func, data, 0));
    });
}

void CommandBuffer::sync() {
    pushOperator([](Stream& stream) {
        stream.sync();
    });
}

void CommandBuffer::pushOperator(std::unique_ptr<Impl::Operator>&& op) {
    mCommandQueue.emplace(std::move(op));
}

void CommandBuffer::pushOperator(std::function<void(Stream&)>&& op) {
    mCommandQueue.emplace(std::make_unique<FunctionOperator>(*mResourceManager,
                                                             std::move(op)));
}

ResourceManager& CommandBuffer::getResourceManager() {
    return *mResourceManager;
}

std::unique_ptr<Task> CommandBuffer::bindStream(Stream& stream, std::shared_ptr<Impl::TaskState> promise) {
    return std::make_unique<Task>(stream, std::move(mResourceManager),
        mCommandQueue,std::move(promise));
}

ResourceInstance& ResourceManager::getResource(const ID id) {
    return *mResources.find(id)->second;
}

void ResourceManager::setStream(cudaStream_t stream) {
    mStream = stream;
}

cudaStream_t ResourceManager::getStream() const {
    return mStream;
}

void ResourceManager::gc(const ID time) {
    std::vector<ID> list;
    for (auto&& x : mResources)
        if (x.second->shouldRelease(time))
            list.emplace_back(x.first);
    for (auto&& id : list)
        mResources.erase(id);
}

CommandBuffer::CommandBuffer(): mResourceManager(std::make_unique<ResourceManager>()) {}

DispatchSystem::StreamInfo& DispatchSystem::getStream() {
    return *std::min_element(mStreams.begin(), mStreams.end());
}

DispatchSystem::DispatchSystem(const size_t size, CommandBufferQueue& queue)
    : mStreams(size), mQueue(queue) {}

void DispatchSystem::update() {
    auto&& stream = getStream();
    if (stream.free()) {
        auto task = mQueue.getTask();
        if (task.first)stream.set(std::move(task));
        else std::this_thread::yield();
    }
    stream.update(Clock::now());
}

Future::Future(std::shared_ptr<Impl::TaskState> promise): mPromise(std::move(promise)) {}

bool Future::finished() const {
    return mPromise->isReleased;
}

FunctionOperator::FunctionOperator(ResourceManager& manager,
                                   std::function<void(Stream&)>&& closure): Operator(manager), mClosure(closure) {}

void FunctionOperator::emit(Stream& stream) {
    mClosure(stream);
}

DispatchSystem::StreamInfo::StreamInfo(): mLast(Clock::now()) {}

bool DispatchSystem::StreamInfo::free() const {
    return mTask == nullptr;
}

void DispatchSystem::StreamInfo::set(CommandBufferQueue::UnboundTask&& task) {
    mTask = task.second->bindStream(mStream, std::move(task.first));
}

void DispatchSystem::StreamInfo::update(const Clock::time_point point) {
    mPool.erase(std::remove_if(mPool.begin(), mPool.end(),
                               [](auto&& task) {
                                   return task->isDone();
                               }), mPool.end());
    if (mTask) {
        mTask->update();
        if (mTask->finished())
            mPool.emplace_back(std::move(mTask));
    }
    mLast = point;
}

bool DispatchSystem::StreamInfo::operator<(const StreamInfo& rhs) const {
    return mLast < rhs.mLast;
}

ResourceInstance::ResourceInstance(): mEnd(Impl::getPID()) {}

bool ResourceInstance::shouldRelease(const ID current) const {
    return current > mEnd;
}

void CommandBufferQueue::submit(std::shared_ptr<Impl::TaskState> promise, 
    std::unique_ptr<CommandBuffer> buffer) {
    std::lock_guard<std::mutex> guard(mMutex);
    mQueue.emplace(std::move(promise),std::move(buffer));
}

CommandBufferQueue::UnboundTask CommandBufferQueue::getTask() {
    if (mQueue.empty())return {};
    std::lock_guard<std::mutex> guard(mMutex);
    auto ptr = std::move(mQueue.front());
    mQueue.pop();
    return ptr;
}

size_t CommandBufferQueue::size() const {
    return mQueue.size();
}

void CommandBufferQueue::clear() {
    decltype(mQueue) empty;
    mQueue.swap(empty);
}

Task::Task(Stream & stream, std::unique_ptr<ResourceManager> manager,
    std::queue<std::unique_ptr<Impl::Operator>>& commandQueue,
           std::shared_ptr<Impl::TaskState> promise)
    :mResourceManager(std::move(manager)),mLast(0),mUpdated(0),
    mPromise(std::move(promise)), mStream(stream) {
    mCommandQueue.swap(commandQueue);
    mResourceManager->setStream(mStream.get());
}

Task::~Task() {
    mPromise->isReleased = true;
}

namespace Impl {
    static void CUDART_CB updateLast(cudaStream_t, cudaError_t, void* userData) {
        const auto info = reinterpret_cast<std::pair<Task*, ID>*>(userData);
        info->first->mLast = info->second;
        delete info;
    }

    static void CUDART_CB streamCallback(cudaStream_t, cudaError_t, void* userData) {
        reinterpret_cast<TaskState*>(userData)->isDone = true;
    }
}

void Task::update() {
    if (!mCommandQueue.empty()) {
        auto&& command = mCommandQueue.front();
        command->emit(mStream);
        if ((mCommandQueue.size() & 0b111) == 0) {
            const auto ptr = new std::pair<Task*, ID>(this, command->getID());
            checkError(cudaStreamAddCallback(mStream.get(), Impl::updateLast, ptr, 0));
        }
        mCommandQueue.pop();
        if (mCommandQueue.empty())
            checkError(cudaStreamAddCallback(mStream.get(), Impl::streamCallback, mPromise.get(), 0));
        if (mLast != mUpdated) {
            mResourceManager->gc(mLast);
            mLast = mUpdated;
        }
    }
}

bool Task::finished() const {
    return mCommandQueue.empty();
}

bool Task::isDone() const {
    return mPromise->isDone;
}
