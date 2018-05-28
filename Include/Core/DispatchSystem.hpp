#pragma once
#include <Core/Pipeline.hpp>
#include <map>
#include <queue>
#include <type_traits>

class CommandBuffer;
class ResourceRecycler;
class ResourceManager;
class StreamContext;

namespace Impl {
    class Operator;
}

namespace Impl {
    struct TaskState final {
        bool isLaunched;
        Event event;
        TaskState() : isLaunched(false) {}
        bool isDone() const;
    };
}

class Future final {
private:
    std::shared_ptr<Impl::TaskState> mPromise;
public:
    explicit Future(std::shared_ptr<Impl::TaskState> promise);
    void sync() const;
    bool finished() const;
};

class Task final : Uncopyable {
private:
    std::unique_ptr<ResourceManager> mResourceManager;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
    std::shared_ptr<Impl::TaskState> mPromise;
    Stream& mStream;
public:
    Task(StreamContext& stream, std::unique_ptr<ResourceManager> manager,
        std::queue<std::unique_ptr<Impl::Operator>>& commandQueue,
        std::shared_ptr<Impl::TaskState> promise);
    bool update();
    bool isDone() const;
};

namespace moodycamel {
    struct ConcurrentQueueDefaultTraits;
    template <typename T, typename Traits = ConcurrentQueueDefaultTraits>
    class ConcurrentQueue;
}

class CommandBufferQueue final : Uncopyable {
public:
    using UnboundTask = std::pair<std::shared_ptr<Impl::TaskState>, std::unique_ptr<CommandBuffer>>;
private:
    std::unique_ptr<moodycamel::ConcurrentQueue<UnboundTask>> mQueue;
public:
    CommandBufferQueue();
    ~CommandBufferQueue();
    void submit(std::shared_ptr<Impl::TaskState> promise, std::unique_ptr<CommandBuffer> buffer);
    UnboundTask getTask();
    size_t size() const;
    void clear();
};

class StreamContext final : Uncopyable {
private:
    Stream mStream;
    std::unique_ptr<Task> mTask;
    std::vector<std::pair<uint64_t, std::unique_ptr<Task>>> mPool;
    std::map<size_t, std::unique_ptr<ResourceRecycler>> mRecyclers;
    Clock::time_point mLast;
    uint64_t mTaskCount;
    void setCurrent(ResourceRecycler& recycler);
public:
    StreamContext();
    ~StreamContext();

    template <typename Recycler>
    Recycler& getRecycler() {
        const auto tid = typeid(Recycler).hash_code();
        auto it = mRecyclers.find(tid);
        if (it == mRecyclers.end()) {
            std::unique_ptr<ResourceRecycler> ptr = std::make_unique<Recycler>();
            setCurrent(*ptr);
            it = mRecyclers.emplace(tid, std::move(ptr)).first;
        }
        return dynamic_cast<Recycler&>(*it->second);
    }

    bool free() const;
    void set(CommandBufferQueue::UnboundTask&& task);
    Stream& getStream();
    void update(Clock::time_point point);
    bool operator<(const StreamContext& rhs) const;
};

class DispatchSystem final : Uncopyable {
private:
    StreamContext& getStream();
    std::vector<StreamContext> mStreams;
    CommandBufferQueue& mQueue;
    bool mYield;
    size_t mIndex;
public:
    DispatchSystem(CommandBufferQueue& queue, size_t index, bool yield);
    size_t getId() const;
    void update();
};
