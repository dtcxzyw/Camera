#include <Base/DispatchSystem.hpp>
#include <Base/Constant.hpp>
#include <algorithm>
#include <utility>
#include <Base/Environment.hpp>

namespace Impl {

    class L1GlobalMemoryPool final : public ResourceRecycler {
    private:
        std::vector<UniqueMemory> mPool[41];
        size_t mMaxUse[41]{}, mCurrent[41]{};
    public:
        void registerAlloc(const size_t size) {
            if (size) {
                const auto level = calcSizeLevel(size);
                ++mCurrent[level];
                mMaxUse[level] = std::max(mMaxUse[level], mCurrent[level]);
            }
        }

        void registerFree(const size_t size) {
            if (size)--mCurrent[calcSizeLevel(size)];
        }

        UniqueMemory alloc(const size_t size) {
            if (size == 0)return nullptr;
            const auto level = calcSizeLevel(size);
            if (mPool[level].empty())return allocGlobalMemory(size);
            auto ptr = std::move(mPool[level].back());
            mPool[level].pop_back();
            return ptr;
        }

        void free(UniqueMemory ptr, const size_t size) {
            if (size == 0)return;
            const auto level = calcSizeLevel(size);
            if (mPool[level].size() < mMaxUse[level])
                mPool[level].emplace_back(std::move(ptr));
        }
    };

    DeviceMemory::DeviceMemory(ResourceManager& manager, const size_t size, const MemoryType type)
        : Resource(manager), mSize(size), mType(type) {
        if (mType == MemoryType::Global)
            manager.getRecycler<L1GlobalMemoryPool>().registerAlloc(mSize);
    }

    DeviceMemory::~DeviceMemory() {
        if (mType == MemoryType::Global)
            addInstance(std::make_unique<GlobalMemory>(mManager, mSize));
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

    void DeviceMemoryInstance::memset(int, Stream&) {
        throw std::logic_error("This memory doesn't support memset.");
    }

    bool GlobalMemory::hasRecycler() const {
        return true;
    }

    GlobalMemory::GlobalMemory(ResourceManager& manager, const size_t size)
        : DeviceMemoryInstance(size), mPool(manager.getRecycler<L1GlobalMemoryPool>()) {
        mPool.registerFree(mSize);
    }

    GlobalMemory::~GlobalMemory() {
        mPool.free(std::move(mMemory), mSize);
    }

    void* GlobalMemory::get() {
        if (!mMemory)mMemory = mPool.alloc(mSize);
        return mMemory.get();
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

    DeviceMemoryInstance& Operator::getMemory(const Id id) const {
        return dynamic_cast<DeviceMemoryInstance&>(mManager.getResource(id));
    }

    Operator::Operator(ResourceManager& manager)
        : mManager(manager), mId(manager.getOperatorPid()) {}

    Id Operator::getId() const {
        return mId;
    }

    Memset::Memset(ResourceManager& manager, const Id memoryID, const int mask)
        : Operator(manager), mMemoryId(memoryID), mMask(mask) {}

    void Memset::emit(Stream& stream) {
        getMemory(mMemoryId).memset(mMask, stream);
    }

    Memcpy::Memcpy(ResourceManager& manager, const Id dst
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

    void CastTag::get(ResourceManager& manager, const Id id, void* ptr) {
        manager.getResource(id).getRes(ptr, manager.getStream());
    }

    DMRef::DMRef(const std::shared_ptr<DeviceMemory>& ref): ResourceRef(ref) {}

    size_t DMRef::size() const {
        return dynamic_cast<DeviceMemory&>(*mRef).size();
    }

    void LaunchSize::download(unsigned int& dst, CommandBuffer& buffer) const {
        auto id = mHelper;
        auto&& manager = buffer.getResourceManager();
        buffer.pushOperator([id,&manager,&dst](Id,ResourceManager&,Stream& stream) {
            checkError(cudaMemcpyAsync(&dst, cast(id, manager),
                                       sizeof(unsigned int), cudaMemcpyDeviceToHost, stream.get()));
        });
    }
}

void ResourceManager::registerResource(Id id, std::unique_ptr<ResourceInstance>&& instance) {
    ++mRegisteredResourceCount;
    #ifdef CAMERA_RESOURCE_CHECK
    mUnknownResource.erase(id);
    #endif
    mResources.emplace(id, std::make_pair(mOperatorCount, std::move(instance)));
}

void CommandBuffer::memset(Impl::DMRef& memory, int mark) {
    mCommandQueue.emplace(std::make_unique<Impl::Memset>(*mResourceManager,
                                                         memory.getId(), mark));
}

void CommandBuffer::memcpy(Impl::DMRef& dst,
                           std::function<void(std::function<void(const void*)>)>&& src) {
    mCommandQueue.emplace(std::make_unique<Impl::Memcpy>(*mResourceManager,
                                                         dst.getId(), std::move(src)));
}

namespace Impl {
    struct CallbackInfo final {
        Id id;
        ResourceManager& manager;
        std::function<void()> func;

        CallbackInfo(const Id time, ResourceManager& resManager, std::function<void()> callable):
            id(time), manager(resManager), func(std::move(callable)) {}
    };

    static void CUDART_CB streamCallback(cudaStream_t, cudaError_t, void* userData) {
        std::unique_ptr<CallbackInfo> ptr(reinterpret_cast<CallbackInfo*>(userData));
        ptr->func();
        ptr->manager.syncPoint(ptr->id);
    }
}

void CommandBuffer::addCallback(const std::function<void()>& func) {
    pushOperator([func](Id id,ResourceManager& manager,Stream& stream) {
        const auto data = new Impl::CallbackInfo(id, manager, func);
        checkError(cudaStreamAddCallback(stream.get(), Impl::streamCallback, data, 0));
    });
}

void CommandBuffer::sync() {
    pushOperator([](Id id,ResourceManager& manager,Stream& stream) {
        stream.sync();
        manager.syncPoint(id);
    });
}

void CommandBuffer::pushOperator(std::unique_ptr<Impl::Operator>&& op) {
    mCommandQueue.emplace(std::move(op));
}

void CommandBuffer::pushOperator(std::function<void(Id, ResourceManager&, Stream&)>&& op) {
    mCommandQueue.emplace(std::make_unique<FunctionOperator>(*mResourceManager, std::move(op)));
}

ResourceManager& CommandBuffer::getResourceManager() {
    return *mResourceManager;
}

std::unique_ptr<Task> CommandBuffer::bindStream(Stream& stream,
                                                std::shared_ptr<Impl::TaskState> promise) {
    return std::make_unique<Task>(stream, std::move(mResourceManager),
                                  mCommandQueue, std::move(promise));
}

ResourceInstance& ResourceManager::getResource(const Id id) {
    return *mResources.find(id)->second.second;
}

void ResourceManager::bindStream(cudaStream_t stream) {
    if (mResourceCount != mRegisteredResourceCount)
        throw std::logic_error("Some resources haven't been registered yet.");
    mStream = stream;
}

cudaStream_t ResourceManager::getStream() const {
    return mStream;
}

void ResourceManager::gc(const Id time) {
    std::vector<Id> list;
    for (auto&& x : mResources)
        if (mSyncPoint >= x.second.first || (time >= x.second.first && x.second.second->hasRecycler()))
            list.emplace_back(x.first);
    for (auto&& id : list)
        mResources.erase(id);
}

Id ResourceManager::allocResource() {
    ++mResourceCount;
    #ifdef CAMERA_RESOURCE_CHECK
    mUnknownResource.emplace(mResourceCount);
    #endif
    return mResourceCount;
}

Id ResourceManager::getOperatorPid() {
    return ++mOperatorCount;
}

void ResourceManager::syncPoint(const Id time) {
    mSyncPoint = time;
}

CommandBuffer::CommandBuffer(): mResourceManager(std::make_unique<ResourceManager>()) {}

DispatchSystem::StreamInfo& DispatchSystem::getStream() {
    return *std::min_element(mStreams.begin(), mStreams.end());
}

namespace Impl {
    static size_t getAsyncEngineCount() {
        #ifdef CAMERA_SINGLE_STREAM
        return 1;
        #else
        return std::max(1,getDeviceMonitor().getProp().asyncEngineCount);
        #endif
    }
}

DispatchSystem::DispatchSystem(CommandBufferQueue& queue, const size_t index, const bool yield)
    : mStreams(Impl::getAsyncEngineCount()), mQueue(queue), mYield(yield), mIndex(index) {}

size_t DispatchSystem::getId() const {
    return mIndex;
}

void DispatchSystem::update() {
    auto&& stream = getStream();
    if (stream.free()) {
        using namespace std::chrono_literals;
        auto task = mQueue.getTask();
        if (task.first)stream.set(std::move(task));
        else {
            #ifdef CAMERA_HUNGRY_REPORT
            printf("DispatchSystem %u is hungry!\n", static_cast<unsigned int>(mIndex));
            #endif
            if (mYield)std::this_thread::sleep_for(1ms);
        }
    }
    stream.update(Clock::now());
}

Future::Future(std::shared_ptr<Impl::TaskState> promise): mPromise(std::move(promise)) {}

void Future::sync() {
    auto&& env=Environment::get();
    while (!mPromise->isLaunched)env.yield();
    if (env.isMainThread()) {
        while (!finished())env.yield();
    }
    else mPromise->event.sync();
}

bool Future::finished() const {
    return mPromise->isLaunched && mPromise->event.query() == cudaSuccess;
}

FunctionOperator::FunctionOperator(ResourceManager& manager,
                                   std::function<void(Id, ResourceManager&, Stream&)>&& closure)
    : Operator(manager), mClosure(closure) {}

void FunctionOperator::emit(Stream& stream) {
    mClosure(getId(), mManager, stream);
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
    if (mTask && mTask->update())
        mPool.emplace_back(std::move(mTask));
    mLast = point;
}

bool DispatchSystem::StreamInfo::operator<(const StreamInfo& rhs) const {
    return mLast < rhs.mLast;
}

bool ResourceInstance::hasRecycler() const {
    return false;
}

void CommandBufferQueue::submit(std::shared_ptr<Impl::TaskState> promise,
                                std::unique_ptr<CommandBuffer> buffer) {
    std::lock_guard<std::mutex> guard(mMutex);
    mQueue.emplace(std::move(promise), std::move(buffer));
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

Task::Task(Stream& stream, std::unique_ptr<ResourceManager> manager,
           std::queue<std::unique_ptr<Impl::Operator>>& commandQueue,
           std::shared_ptr<Impl::TaskState> promise): mResourceManager(std::move(manager)),
    mPromise(std::move(promise)), mStream(stream) {
    mCommandQueue.swap(commandQueue);
    mResourceManager->bindStream(mStream.get());
}

bool Task::update() {
    if (!mCommandQueue.empty()) {
        auto&& command = mCommandQueue.front();
        #ifdef CAMERA_SYNC
        Event begin(true);
        begin.bind(mStream);
        #endif
        command->emit(mStream);
        #ifdef CAMERA_SYNC
        Event end(true);
        end.bind(mStream);
        end.sync();
        printf("operator %u:%.2f ms\n", command->getId(), end - begin);
        #endif
        mResourceManager->gc(command->getId());
        mCommandQueue.pop();
    }
    if (mCommandQueue.empty()) {
        mPromise->event.bind(mStream);
        mPromise->isLaunched = true;
        return true;
    }
    return false;
}

bool Task::isDone() const {
    return mPromise->event.query() == cudaSuccess;
}
