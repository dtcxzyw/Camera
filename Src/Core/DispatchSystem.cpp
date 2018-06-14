#include <Core/DispatchSystem.hpp>
#include <algorithm>
#include <utility>
#include <Core/Environment.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/IncludeBegin.hpp>
#include <concurrentqueue.h>
#include <Core/IncludeEnd.hpp>

StreamContext& DispatchSystem::getStream() {
    return *std::min_element(mStreams.begin(), mStreams.end());
}

namespace Impl {
    static size_t getAsyncEngineCount() {
        #ifdef CAMERA_SINGLE_STREAM
        return 1;
        #else
        return std::max(1, DeviceMonitor::get().getProp().asyncEngineCount);
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
            printf("DispatchSystem %u is hungry!\n", static_cast<uint32_t>(mIndex));
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

void StreamContext::setCurrent(ResourceRecycler& recycler) {
    recycler.setCurrent(mTaskCount);
}

StreamContext::StreamContext(): mLast(Clock::now()), mTaskCount(0) {}

StreamContext::~StreamContext() {
    mTask.reset();
    mRecyclers.clear();
}

bool StreamContext::free() const {
    return mTask == nullptr;
}

void StreamContext::set(CommandBufferQueue::UnboundTask&& task) {
    ++mTaskCount;
    for (auto&& recycler : mRecyclers)
        recycler.second->setCurrent(mTaskCount);
    mTask = task.second->bindStream(*this, std::move(task.first));
}

Stream& StreamContext::getStream() {
    return mStream;
}

void StreamContext::update(const Clock::time_point point) {
    mPool.erase(std::remove_if(mPool.begin(), mPool.end(),
        [this](auto&& task) {
            if (task.second->isDone()) {
                for (auto&& recycler : mRecyclers)
                    recycler.second->gc(task.first);
                return true;
            }
            return false;
        }), mPool.end());
    if (mTask && mTask->update())
        mPool.emplace_back(mTaskCount, std::move(mTask));
    mLast = point;
}

bool StreamContext::operator<(const StreamContext& rhs) const {
    return mLast < rhs.mLast;
}

bool ResourceInstance::canBeRecycled() const {
    return false;
}

void ResourceInstance::bindStream(StreamContext&) {}

CommandBufferQueue::CommandBufferQueue() {
    mQueue = std::make_unique<moodycamel::ConcurrentQueue<UnboundTask>>();
}

CommandBufferQueue::~CommandBufferQueue() {
    mQueue.reset();
}

void CommandBufferQueue::submit(std::shared_ptr<Impl::TaskState> promise,
    std::unique_ptr<CommandBuffer> buffer) {
    mQueue->enqueue({std::move(promise), std::move(buffer)});
}

CommandBufferQueue::UnboundTask CommandBufferQueue::getTask() {
    UnboundTask res;
    mQueue->try_dequeue(res);
    return res;
}

size_t CommandBufferQueue::size() const {
    return mQueue->size_approx();
}

void CommandBufferQueue::clear() {
    decltype(mQueue) empty;
    mQueue.swap(empty);
}

Task::Task(StreamContext& stream, std::unique_ptr<ResourceManager> manager,
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
