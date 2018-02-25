#include <Base/DispatchSystem.hpp>
#include <Base/Constant.hpp>
#include <algorithm>
#include <utility>

namespace Impl {

    DeviceMemory::DeviceMemory(ResourceManager& manager, const size_t size, const MemoryType type)
        : Resource(manager), mSize(size), mType(type) {}

    DeviceMemory::~DeviceMemory() {
        if (mType == MemoryType::Global)
            addInstance(std::make_unique<GlobalMemory>(mManager,mSize));
        else
            addInstance(std::make_unique<ConstantMemory>(mSize));
    }

    size_t DeviceMemory::size() const {
        return mSize;
    }

    void DeviceMemoryInstance::getRes(void* res, cudaStream_t) {
        *reinterpret_cast<void**>(res) = get();
    }

    DeviceMemoryInstance::DeviceMemoryInstance(const size_t size):mSize(size) {}

    void DeviceMemoryInstance::memset(int, Stream&) {
        throw std::logic_error("This memory doesn't support memset.");
    }

    class L1GlobalMemoryPool final:public ResourceRecycler {
    private:
        std::vector<UniqueMemory> mPool[41];
    public:
        UniqueMemory alloc(const size_t size) {
            if (size == 0)return nullptr;
            const auto level = calcSizeLevel(size);
            if(mPool[level].empty())return allocGlobalMemory(size);
            auto ptr = std::move(mPool[level].back());
            mPool[level].pop_back();
            return ptr;
        }
        void free(UniqueMemory ptr,const size_t size) {
            if(size==0)return;
            const auto level = calcSizeLevel(size);
            mPool[level].emplace_back(std::move(ptr));
        }
    };

    bool GlobalMemory::hasRecycler() const {
        return true;
    }

    GlobalMemory::GlobalMemory(ResourceManager& manager,const size_t size)
        : DeviceMemoryInstance(size),mPool(manager.getRecycler<L1GlobalMemoryPool>()) {}

    GlobalMemory::~GlobalMemory() {
        mPool.free(std::move(mMemory),mSize);
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

    DeviceMemoryInstance& Operator::getMemory(const ID id) const {
        return dynamic_cast<DeviceMemoryInstance&>(mManager.getResource(id));
    }

    Operator::Operator(ResourceManager& manager)
        : mManager(manager), mID(manager.getOperatorPID()) {}

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

    void LaunchSize::download(unsigned int& dst, CommandBuffer& buffer) const {
        auto id = mHelper;
        auto&& manager = buffer.getResourceManager();
        buffer.pushOperator([id,&manager,&dst](ID,ResourceManager&,Stream& stream) {
            checkError(cudaMemcpyAsync(&dst, cast(id, manager),
                                       sizeof(unsigned int), cudaMemcpyDeviceToHost, stream.get()));
        });
    }
}

void ResourceManager::registerResource(ID id, std::unique_ptr<ResourceInstance>&& instance) {
    ++mRegisteredResourceCount;
    mResources.emplace(id,std::make_pair(mOperatorCount,std::move(instance)));
}

void CommandBuffer::memset(Impl::DMRef& memory, int mark) {
    mCommandQueue.emplace(std::make_unique<Impl::Memset>(*mResourceManager,
                                                         memory.getID(), mark));
}

void CommandBuffer::memcpy(Impl::DMRef& dst,
                           std::function<void(std::function<void(const void*)>)>&& src) {
    mCommandQueue.emplace(std::make_unique<Impl::Memcpy>(*mResourceManager,
                                                         dst.getID(), std::move(src)));
}

namespace Impl {
    struct CallbackInfo final {
        ID id;
        ResourceManager& manager;
        std::function<void()> func;
        CallbackInfo(const ID time,ResourceManager& resManager, std::function<void()> callable):
            id(time),manager(resManager),func(std::move(callable)){}
    };

    static void CUDART_CB streamCallback(cudaStream_t, cudaError_t, void* userData) {
        auto ptr = reinterpret_cast<CallbackInfo*>(userData);
        ptr->func();
        ptr->manager.syncPoint(ptr->id);
        delete ptr;
    }
}

void CommandBuffer::addCallback(const std::function<void()>& func) {
    pushOperator([func](ID id,ResourceManager& manager,Stream& stream) {
        const auto data = new Impl::CallbackInfo(id,manager,func);
        checkError(cudaStreamAddCallback(stream.get(), Impl::streamCallback, data, 0));
    });
}

void CommandBuffer::sync() {
    pushOperator([](ID id,ResourceManager& manager,Stream& stream) {
        stream.sync();
        manager.syncPoint(id);
    });
}

void CommandBuffer::pushOperator(std::unique_ptr<Impl::Operator>&& op) {
    mCommandQueue.emplace(std::move(op));
}

void CommandBuffer::pushOperator(std::function<void(ID,ResourceManager&, Stream&)>&& op) {
    mCommandQueue.emplace(std::make_unique<FunctionOperator>(*mResourceManager,std::move(op)));
}

ResourceManager& CommandBuffer::getResourceManager() {
    return *mResourceManager;
}

std::unique_ptr<Task> CommandBuffer::bindStream(Stream& stream, std::shared_ptr<Impl::TaskState> promise) {
    return std::make_unique<Task>(stream, std::move(mResourceManager),
                                  mCommandQueue, std::move(promise));
}

ResourceInstance& ResourceManager::getResource(const ID id) {
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

void ResourceManager::gc(const ID time) {
    std::vector<ID> list;
    for (auto&& x : mResources)
        if (mSyncPoint>=x.second.first || (time>= x.second.first && x.second.second->hasRecycler()))
            list.emplace_back(x.first);
    for (auto&& id : list)
        mResources.erase(id);
}

ID ResourceManager::allocResource() {
    return ++mResourceCount;
}

ID ResourceManager::getOperatorPID() {
    return ++mOperatorCount;
}

void ResourceManager::syncPoint(const ID time) {
    mSyncPoint = time;
}

CommandBuffer::CommandBuffer(): mResourceManager(std::make_unique<ResourceManager>()) {}

DispatchSystem::StreamInfo& DispatchSystem::getStream() {
    return *std::min_element(mStreams.begin(), mStreams.end());
}

namespace Impl {
    static size_t getAsyncEngineCount() {
        int device;
        checkError(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        checkError(cudaGetDeviceProperties(&prop, device));
        return std::max(1, prop.asyncEngineCount);
    }
}

DispatchSystem::DispatchSystem(CommandBufferQueue& queue)
    : mStreams(Impl::getAsyncEngineCount()), mQueue(queue) {}

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
    std::function<void(ID,ResourceManager&, Stream&)>&& closure)
    : Operator(manager), mClosure(closure) {}

void FunctionOperator::emit(Stream& stream) {
    mClosure(getID(),mManager,stream);
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
           std::shared_ptr<Impl::TaskState> promise)
    : mResourceManager(std::move(manager)),
    mPromise(std::move(promise)), mStream(stream) {
    mCommandQueue.swap(commandQueue);
    mResourceManager->bindStream(mStream.get());
}

Task::~Task() {
    mPromise->isReleased = true;
}

namespace Impl {
    static void CUDART_CB endCallback(cudaStream_t, cudaError_t, void* userData) {
        reinterpret_cast<TaskState*>(userData)->isDone = true;
    }
}

void Task::update() {
    if (!mCommandQueue.empty()) {
        auto&& command = mCommandQueue.front();
        command->emit(mStream);
        mResourceManager->gc(command->getID());
        mCommandQueue.pop();
        if (mCommandQueue.empty())
            checkError(cudaStreamAddCallback(mStream.get(), Impl::endCallback, mPromise.get(), 0));
    }
}

bool Task::finished() const {
    return mCommandQueue.empty();
}

bool Task::isDone() const {
    return mPromise->isDone;
}
