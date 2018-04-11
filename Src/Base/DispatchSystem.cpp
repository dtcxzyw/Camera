#include <Base/DispatchSystem.hpp>
#include <Base/Constant.hpp>
#include <algorithm>
#include <utility>
#include <Base/Environment.hpp>

namespace Impl {

    class L1GlobalMemoryPool final : public ResourceRecycler {
    private:
        std::vector<UniqueMemory> mPool[41];
    public:
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
            mPool[level].emplace_back(std::move(ptr));
        }
    };

    void DeviceMemoryInstance::getRes(void* res, cudaStream_t) {
        *reinterpret_cast<void**>(res) = get();
    }

    DeviceMemoryInstance::DeviceMemoryInstance(const size_t size): mSize(size) {}

    void DeviceMemoryInstance::memset(int, size_t, size_t, Stream&) {
        throw std::logic_error("This memory doesn't support memset.");
    }

    bool GlobalMemory::canBeRecycled() const {
        return !mOnRelease;
    }

    GlobalMemory::GlobalMemory(const size_t size,
        MemoryReleaseFunction onRelease)
        : DeviceMemoryInstance(size), mPool(nullptr), mOnRelease(std::move(onRelease)) {}

    void GlobalMemory::bindStream(StreamInfo& info) {
        mPool = &info.getRecycler<L1GlobalMemoryPool>();
    }

    GlobalMemory::~GlobalMemory() {
        if (mOnRelease)mOnRelease(std::move(mMemory), mSize);
        else if(mPool)mPool->free(std::move(mMemory), mSize);
    }

    void* GlobalMemory::get() {
        if (!mMemory)mMemory = mPool->alloc(mSize);
        return mMemory.get();
    }

    void GlobalMemory::set(const void* src, const size_t begin, const size_t end, Stream& stream) {
        checkError(cudaMemcpyAsync(static_cast<unsigned char*>(get()) + begin, src, end - begin,
            cudaMemcpyDefault, stream.get()));
    }

    void GlobalMemory::memset(const int mask, const size_t begin, const size_t end, Stream& stream) {
        checkError(cudaMemsetAsync(static_cast<unsigned char*>(get()) + begin, mask, end - begin, 
            stream.get()));
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

    void ConstantMemory::set(const void* src, const size_t begin, const size_t end, Stream& stream) {
        constantSet(static_cast<unsigned char*>(get()) + begin, src,
            static_cast<unsigned int>(end - begin), stream.get());
    }

    AllocatedMemory::AllocatedMemory(const MemorySpan<unsigned char>& ref)
        : DeviceMemoryInstance(ref.size()), mRef(ref) {}

    void* AllocatedMemory::get() {
        return mRef.begin();
    }

    void AllocatedMemory::set(const void* src, const size_t begin, const size_t end, Stream& stream) {
        checkError(cudaMemcpyAsync(mRef.begin() + begin, src, end - begin, cudaMemcpyDefault,
            stream.get()));
    }

    void AllocatedMemory::memset(const int mask, const size_t begin, const size_t end, Stream& stream) {
        stream.memset(mRef.subSpan(begin, end), mask);
    }

    DeviceMemoryInstance& Operator::getMemory(const Id id) const {
        return dynamic_cast<DeviceMemoryInstance&>(mManager.getResource(id));
    }

    Operator::Operator(ResourceManager& manager)
        : mManager(manager), mId(manager.getOperatorPid()) {}

    Id Operator::getId() const {
        return mId;
    }

    FunctionOperator::FunctionOperator(ResourceManager& manager,
        std::function<void(Id, ResourceManager&, Stream&)>&& closure)
        : Operator(manager), mClosure(closure) {}

    void FunctionOperator::emit(Stream& stream) {
        mClosure(getId(), mManager, stream);
    }

    Memset::Memset(ResourceManager& manager, const SpanHelper<unsigned char> span, const int mask)
        : Operator(manager), mSpan(span), mMask(mask) {}

    void Memset::emit(Stream& stream) {
        getMemory(mSpan.getId()).memset(mMask, mSpan.begin(), mSpan.end(), stream);
    }

    Memcpy::Memcpy(ResourceManager& manager, const SpanHelper<unsigned char> dst
        , std::function<void(std::function<void(const void*)>)>&& src)
        : Operator(manager), mDst(dst), mSrc(src) {}

    void Memcpy::emit(Stream& stream) {
        mSrc([this, &stream](auto ptr) {
            getMemory(mDst.getId()).set(ptr, mDst.begin(), mDst.end(), stream);
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

    LaunchSize::LaunchSize(const Span<unsigned>& ptr) : mHelper(ptr), mRef(ptr) {}

    SpanHelper<unsigned int> LaunchSize::get() const {
        return mHelper;
    }

    void LaunchSize::download(unsigned int& dst, CommandBuffer& buffer) const {
        auto&& manager = buffer.getResourceManager();
        auto id = mHelper;
        buffer.pushOperator([id, &manager, &dst](Id, ResourceManager&, Stream& stream) {
            checkError(cudaMemcpyAsync(&dst, cast(id, manager),
                sizeof(unsigned int), cudaMemcpyDeviceToHost, stream.get()));
        });
    }

    DeviceMemoryDesc::DeviceMemoryDesc(ResourceManager& manager, size_t size)
        : Resource<void*>(manager), mSize(size) {}

    size_t DeviceMemoryDesc::size() const {
        return mSize;
    }

    GlobalMemoryDesc::GlobalMemoryDesc(ResourceManager& manager, const size_t size,
        MemoryReleaseFunction onRelease) : DeviceMemoryDesc(manager, size),
        mOnRelease(std::move(onRelease)) {}

    GlobalMemoryDesc::~GlobalMemoryDesc() {
        addInstance(std::make_unique<GlobalMemory>(mSize, std::move(mOnRelease)));
    }

    ConstantMemoryDesc::ConstantMemoryDesc(ResourceManager& manager, const size_t size)
        : DeviceMemoryDesc(manager, size) {}

    ConstantMemoryDesc::~ConstantMemoryDesc() {
        addInstance(std::make_unique<ConstantMemory>(mSize));
    }

    AllocatedMemoryDesc::AllocatedMemoryDesc(ResourceManager& manager,
        const MemorySpan<unsigned char>& ref) : DeviceMemoryDesc(manager, ref.size()), mRef(ref) {}

    AllocatedMemoryDesc::~AllocatedMemoryDesc() {
        addInstance(std::make_unique<AllocatedMemory>(mRef));
    }

}

void ResourceManager::registerResource(Id id, std::unique_ptr<ResourceInstance>&& instance) {
    #ifdef CAMERA_RESOURCE_CHECK
    mUnknownResource.erase(id);
    #endif
    mResources.emplace(id, std::make_pair(mOperatorCount, std::move(instance)));
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
    mCommandQueue.emplace(std::make_unique<Impl::FunctionOperator>(*mResourceManager,
        std::move(op)));
}

ResourceManager& CommandBuffer::getResourceManager() {
    return *mResourceManager;
}

std::unique_ptr<Task> CommandBuffer::bindStream(StreamInfo& stream,
    std::shared_ptr<Impl::TaskState> promise) {
    return std::make_unique<Task>(stream, std::move(mResourceManager),
        mCommandQueue, std::move(promise));
}

ResourceInstance& ResourceManager::getResource(const Id id) {
    return *mResources.find(id)->second.second;
}

void ResourceManager::bindStream(StreamInfo& stream) {
#ifdef CAMERA_RESOUREC_CHECK
    if (mResourceCount != mResources.size())
        throw std::logic_error("Some resources haven't been registered yet.");
#endif
    for (auto&& res : mResources)
        res.second.second->bindStream(stream);
    mStream = stream.getStream().get();
}

cudaStream_t ResourceManager::getStream() const {
    return mStream;
}

void ResourceManager::gc(const Id time) {
    std::vector<Id> list;
    for (auto&& x : mResources)
        if (mSyncPoint >= x.second.first || (time >= x.second.first && x.second.second->canBeRecycled()))
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

void CommandBuffer::memset(const Span<unsigned char>& memory, const int mask) {
    mCommandQueue.emplace(std::make_unique<Impl::Memset>(*mResourceManager,
        Impl::castId(memory), mask));
}

void CommandBuffer::memcpy(const Span<unsigned char>& dst,
    std::function<void(std::function<void(const void*)>)>&& src) {
    mCommandQueue.emplace(std::make_unique<Impl::Memcpy>(*mResourceManager,
        Impl::castId(dst), std::move(src)));
}

void CommandBuffer::memcpy(const Span<unsigned char>& dst,
    const MemorySpan<unsigned char>& src) {
    memcpy(dst, [src](auto call) {
        call(src.begin());
    });
}

StreamInfo& DispatchSystem::getStream() {
    return *std::min_element(mStreams.begin(), mStreams.end());
}

namespace Impl {
    static size_t getAsyncEngineCount() {
        #ifdef CAMERA_SINGLE_STREAM
        return 1;
        #else
        return std::max(1, getDeviceMonitor().getProp().asyncEngineCount);
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

bool Impl::TaskState::isDone() const {
    return event.query() == cudaSuccess;
}

Future::Future(std::shared_ptr<Impl::TaskState> promise): mPromise(std::move(promise)) {}

void Future::sync() const {
    auto&& env = Environment::get();
    while (!mPromise->isLaunched)env.yield();
    if (env.isMainThread()) {
        while (!finished())env.yield();
    }
    else mPromise->event.sync();
}

bool Future::finished() const {
    return mPromise->isLaunched && mPromise->isDone();
}

StreamInfo::StreamInfo(): mLast(Clock::now()) {}
StreamInfo::~StreamInfo() {
    mTask.reset();
    mRecyclers.clear();
}

bool StreamInfo::free() const {
    return mTask == nullptr;
}

void StreamInfo::set(CommandBufferQueue::UnboundTask&& task) {
    mTask = task.second->bindStream(*this, std::move(task.first));
}

Stream& StreamInfo::getStream() {
    return mStream;
}

void StreamInfo::update(const Clock::time_point point) {
    mPool.erase(std::remove_if(mPool.begin(), mPool.end(),
        [](auto&& task) {
            return task->isDone();
        }), mPool.end());
    if (mTask && mTask->update())
        mPool.emplace_back(std::move(mTask));
    mLast = point;
}

bool StreamInfo::operator<(const StreamInfo& rhs) const {
    return mLast < rhs.mLast;
}

bool ResourceInstance::canBeRecycled() const {
    return false;
}

void ResourceInstance::bindStream(StreamInfo&) {}

void CommandBufferQueue::submit(std::shared_ptr<Impl::TaskState> promise,
    std::unique_ptr<CommandBuffer> buffer) {
    mQueue.enqueue({std::move(promise), std::move(buffer)});
}

CommandBufferQueue::UnboundTask CommandBufferQueue::getTask() {
    UnboundTask res;
    mQueue.try_dequeue(res);
    return res;
}

size_t CommandBufferQueue::size() const {
    return mQueue.size_approx();
}

void CommandBufferQueue::clear() {
    decltype(mQueue) empty;
    mQueue.swap(empty);
}

Task::Task(StreamInfo& stream, std::unique_ptr<ResourceManager> manager,
    std::queue<std::unique_ptr<Impl::Operator>>& commandQueue,
    std::shared_ptr<Impl::TaskState> promise): mResourceManager(std::move(manager)),
    mPromise(std::move(promise)), mStream(stream.getStream()) {
    mCommandQueue.swap(commandQueue);
    mResourceManager->bindStream(stream);
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
        printf("operator %u:%.4f ms\n", command->getId(), end - begin);
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
    return mPromise->isDone();
}
