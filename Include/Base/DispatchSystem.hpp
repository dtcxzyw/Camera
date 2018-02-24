#pragma once
#include <Base/Pipeline.hpp>
#include <map>
#include <queue>
#include <type_traits>
#include <chrono>
#include <tuple>
#include <functional>
#include <atomic>
#include <utility>
#include <mutex>

class CommandBuffer;
class ResourceManager;
using ID = uint64_t;

class ResourceInstance : Uncopyable {
public:
    ResourceInstance() = default;
    virtual ~ResourceInstance() = default;
    virtual bool hasRecycler() const;
    virtual void getRes(void*, cudaStream_t) = 0;
};

template <typename T>
class Resource : Uncopyable {
private:
    const ID mID;
protected:
    ResourceManager& mManager;
    void addInstance(std::unique_ptr<ResourceInstance> instance) const;
public:
    explicit Resource(ResourceManager& manager);
    virtual ~Resource() = default;

    ID getID() const noexcept {
        return mID;
    }
};

namespace Impl {
    struct ResourceTag {};
}

template <typename T>
class ResourceRef : public Impl::ResourceTag {
protected:
    std::shared_ptr<Resource<T>> mRef;
public:
    explicit ResourceRef(std::shared_ptr<Resource<T>> ref) : mRef(std::move(ref)) {}

    void earlyRelease() {
        mRef.reset();
    }

    ID getID() const {
        return mRef->getID();
    }
};

namespace Impl {
    enum class MemoryType {
        Global,
        Constant
    };

    class DeviceMemory final : public Resource<void*> {
    private:
        const size_t mSize;
        MemoryType mType;
    public:
        DeviceMemory(ResourceManager& manager, size_t size, MemoryType type);
        virtual ~DeviceMemory();
        size_t size() const;
    };

    class DeviceMemoryInstance : public ResourceInstance {
    protected:
        const size_t mSize;
        void getRes(void*, cudaStream_t) override;
    public:
        explicit DeviceMemoryInstance(size_t size);
        virtual ~DeviceMemoryInstance() = default;
        virtual void* get() = 0;
        virtual void set(const void* src, Stream& stream) = 0;
        virtual void memset(int mask, Stream& stream);
    };

    class L1GlobalMemoryPool;

    class GlobalMemory final : public DeviceMemoryInstance {
    private:
        UniqueMemory mMemory;
        L1GlobalMemoryPool& mPool;
        bool hasRecycler() const override;
    public:
        GlobalMemory(ResourceManager& manager, size_t size);
        ~GlobalMemory();
        void* get() override;
        void set(const void* src, Stream& stream) override;
        void memset(int mask, Stream& stream) override;
    };

    class ConstantMemory final : public DeviceMemoryInstance {
    private:
        void* mPtr;
    public:
        explicit ConstantMemory(size_t size);
        ~ConstantMemory();
        void* get() override;
        void set(const void* src, Stream& stream) override;
    };

    class DMRef : public ResourceRef<void*> {
    public:
        explicit DMRef(const std::shared_ptr<DeviceMemory>& ref);
        size_t size() const;
    };

    class Operator : Uncopyable {
    protected:
        ResourceManager& mManager;
        ID mID;
        DeviceMemoryInstance& getMemory(ID id) const;
    public:
        explicit Operator(ResourceManager& manager);
        virtual ~Operator() = default;
        ID getID() const;
        virtual void emit(Stream& stream) = 0;
    };

    class Memset final : public Operator {
    private:
        ID mMemoryID;
        int mMask;
    public:
        Memset(ResourceManager& manager, ID memoryID, int mask);
        void emit(Stream& stream) override;
    };

    class Memcpy final : public Operator {
    private:
        ID mDst;
        std::function<void(std::function<void(const void*)>)> mSrc;
    public:
        Memcpy(ResourceManager& manager, ID dst,
               std::function<void(std::function<void(const void*)>)>&& src);
        void emit(Stream& stream) override;
    };
}

class FunctionOperator final : public Impl::Operator {
private:
    std::function<void(ID,ResourceManager&,Stream&)> mClosure;
public:
    FunctionOperator(ResourceManager& manager, 
        std::function<void(ID,ResourceManager&,Stream&)>&& closure);
    void emit(Stream& stream) override;
};

template <typename T>
class MemoryRef final : public Impl::DMRef {
public:
    explicit MemoryRef(const std::shared_ptr<Impl::DeviceMemory>& ref) : DMRef(ref) {}

    size_t size() const {
        return DMRef::size() / sizeof(T);
    }
};

namespace Impl {
    class CastTag {
    protected:
        static void get(ResourceManager& manager, ID id, void* ptr);
    };

    template <typename T>
    class ResourceID final : public CastTag {
    private:
        ID mID;
    public:
        explicit ResourceID(const ID id) : mID(id) {}

        T get(ResourceManager& manager) {
            T res;
            CastTag::get(manager, mID, &res);
            return res;
        }
    };

    template <typename T, typename = std::enable_if_t<!std::is_base_of<ResourceTag, T>::value>>
    T castID(T arg) {
        return arg;
    }

    template <typename T>
    auto castID(const ResourceRef<T>& ref) -> ResourceID<T> {
        return ResourceID<T>{ref.getID()};
    }

    template <typename T>
    auto castID(const MemoryRef<T>& ref) -> ResourceID<T*> {
        return ResourceID<T*>{ref.getID()};
    }

    template <typename T, typename = std::enable_if_t<!std::is_base_of<CastTag, T>::value>>
    T cast(T arg, ResourceManager&) {
        return arg;
    }

    template <typename T, typename = std::enable_if_t<std::is_base_of<CastTag, T>::value>>
    auto cast(T ref, ResourceManager& manager) {
        return ref.get(manager);
    }

    template <typename T, typename... Args>
    class LazyConstructor final : public CastTag {
    private:
        std::tuple<Args...> mArgs;

        template <size_t... I>
        auto constructImpl(ResourceManager& manager, std::index_sequence<I...>) {
            return T{cast(std::get<I>(mArgs), manager)...};
        }

    public:
        explicit LazyConstructor(Args ... args) : mArgs(std::make_tuple(args...)) {}

        auto get(ResourceManager& manager) {
            return constructImpl(manager, std::make_index_sequence<std::tuple_size<decltype(mArgs)>::value>());
        }
    };

    template <typename T>
    class DataPtrHelper final : public CastTag {
    private:
        std::function<T*(ResourceManager&)> mClosure;
    public:
        explicit DataPtrHelper(const MemoryRef<T>& ref) {
            auto rval = castID(ref);
            mClosure = [rval](ResourceManager& manager) {
                return cast(rval, manager);
            };
        }

        explicit DataPtrHelper(const DataViewer<T>& data) {
            auto ptr = data.begin();
            mClosure = [ptr](ResourceManager&) {
                return ptr;
            };
        }

        T* get(ResourceManager& manager) {
            return mClosure(manager);
        }
    };

    template <typename T>
    class DataPtr final {
    private:
        std::function<DataPtrHelper<T>()> mClosure;
        const size_t mSize;
    public:
        DataPtr(const MemoryRef<T>& ref): mSize(ref.size()) {
            mClosure = [ref] {
                return DataPtrHelper<T>(ref);
            };
        }

        DataPtr(const DataViewer<T>& data): mSize(data.size()) {
            mClosure = [data] {
                return DataPtrHelper<T>(data);
            };
        }

        auto get() const {
            return mClosure();
        }

        auto size() const {
            return mSize;
        }
    };

    template <typename T>
    class ValueHelper final : public CastTag {
    private:
        std::function<T(ResourceManager&)> mClosure;
    public:
        template <typename U>
        explicit ValueHelper(const U& val) {
            auto rval = castID(val);
            mClosure = [rval](ResourceManager& manager) {
                return cast(rval, manager);
            };
        }

        T get(ResourceManager& manager) {
            return mClosure(manager);
        }
    };

    template <typename T>
    class Value final {
    private:
        std::function<ValueHelper<T>()> mClosure;
    public:
        template <typename U>
        Value(U val) {
            mClosure = [val] {
                return ValueHelper<T>(val);
            };
        }

        auto get() const {
            return mClosure();
        }
    };

    class LaunchSizeHelper final : public CastTag {
    private:
        std::function<unsigned int*(ResourceManager&)> mClosure;
    public:
        LaunchSizeHelper(const MemoryRef<unsigned int>& ptr, unsigned int off) {
            auto rval = castID(ptr);
            mClosure = [rval,off](ResourceManager& manager) {
                return cast(rval, manager) + off;
            };
        }

        unsigned int* get(ResourceManager& manager) const {
            return mClosure(manager);
        }
    };

    class LaunchSize final {
    private:
        LaunchSizeHelper mHelper;
        MemoryRef<unsigned int> mRef;
    public:
        explicit LaunchSize(const MemoryRef<unsigned int>& ptr, const unsigned int off = 0)
            : mHelper(ptr, off), mRef(ptr) {}

        auto get() const {
            return mHelper;
        }

        void download(std::atomic_uint& dst, CommandBuffer& buffer) const;
    };

    class KernelLaunchDim final : public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template <typename Func, typename... Args>
        KernelLaunchDim(ResourceManager& manager, Func func, const dim3 grid, const dim3 block,
                        Args ... args): Operator(manager) {
            mClosure = [=, &manager](Stream& stream) {
                stream.runDim(func, grid, block, cast(args, manager)...);
            };
        }

        void emit(Stream& stream) override;
    };

    class KernelLaunchLinear final : public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template <typename Func, typename... Args>
        KernelLaunchLinear(ResourceManager& manager, Func func, const size_t size, Args ... args)
            : Operator(manager) {
            mClosure = [=, &manager](Stream& stream) {
                stream.run(func, size, cast(args, manager)...);
            };
        }

        void emit(Stream& stream) override;
    };

}

using Impl::DataPtr;
using Impl::Value;
using Impl::LaunchSize;

namespace Impl {
    struct TaskState final {
        bool isDone;
        bool isReleased;
        TaskState() : isDone(false), isReleased(false) {}
    };
}

class Future final {
private:
    std::shared_ptr<Impl::TaskState> mPromise;
public:
    explicit Future(std::shared_ptr<Impl::TaskState> promise);
    bool finished() const;
};

class ResourceRecycler : Uncopyable {
public:
    virtual ~ResourceRecycler() = default;
};

class ResourceManager final : Uncopyable {
private:
    std::map<ID, std::pair<ID, std::unique_ptr<ResourceInstance>>> mResources;
    cudaStream_t mStream = nullptr;
    ID mResourceCount = 0, mRegisteredResourceCount = 0, mOperatorCount = 0, mSyncPoint = 0;
    std::map<size_t, std::unique_ptr<ResourceRecycler>> mRecyclers;
public:
    void registerResource(ID id, std::unique_ptr<ResourceInstance>&& instance);
    ResourceInstance& getResource(ID id);
    void bindStream(cudaStream_t stream);
    cudaStream_t getStream() const;
    void gc(ID time);
    ID allocResource();
    ID getOperatorPID();
    void syncPoint(ID time);

    template <typename Recycler>
    Recycler& getRecycler() {
        const auto tid = typeid(Recycler).hash_code();
        auto it = mRecyclers.find(tid);
        if (it == mRecyclers.end()) {
            std::unique_ptr<ResourceRecycler> ptr = std::make_unique<Recycler>();
            it = mRecyclers.emplace(tid, std::move(ptr)).first;
        }
        return dynamic_cast<Recycler&>(*it->second);
    }
};

class Task final : Uncopyable {
private:
    std::unique_ptr<ResourceManager> mResourceManager;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
    std::shared_ptr<Impl::TaskState> mPromise;
    Stream& mStream;
public:
    Task(Stream& stream, std::unique_ptr<ResourceManager> manager,
         std::queue<std::unique_ptr<Impl::Operator>>& commandQueue,
         std::shared_ptr<Impl::TaskState> promise);
    ~Task();
    void update();
    bool finished() const;
    bool isDone() const;
};

class CommandBuffer final : Uncopyable {
private:
    std::unique_ptr<ResourceManager> mResourceManager;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
public:
    CommandBuffer();

    template <typename T>
    MemoryRef<T> allocBuffer(const size_t size = 1) {
        return MemoryRef<T>{
            std::make_shared<Impl::DeviceMemory>(*mResourceManager, size * sizeof(T),
                                                 Impl::MemoryType::Global)
        };
    }

    template <typename T>
    MemoryRef<T> allocConstant() {
        return MemoryRef<T>{
            std::make_shared<Impl::DeviceMemory>(*mResourceManager, sizeof(T),
                                                 Impl::MemoryType::Constant)
        };
    }

    void memset(Impl::DMRef& memory, int mark = 0);

    void memcpy(Impl::DMRef& dst, std::function<void(std::function<void(const void*)>)>&& src);

    template <typename T>
    void memcpy(MemoryRef<T>& dst, const DataViewer<T>& src) {
        memcpy(dst, [src](auto call) {
            call(src.begin());
        });
    }

    template <typename T>
    void memcpy(MemoryRef<T>& dst, const MemoryRef<T>& src) {
        const auto id = src.getID();
        memcpy(dst, [id, this](auto call) {
            void* res;
            mResourceManager->getResource(id).getRes(&res, mResourceManager->getStream());
            call(res);
        });
    }

    template <typename Func, typename... Args>
    void runKernelDim(Func func, const dim3 grid, const dim3 block, Args ... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchDim>(*mResourceManager,
                                                                      func, grid, block, Impl::castID(args)...));
    }

    template <typename Func, typename... Args>
    void callKernel(Func func, Args ... args) {
        runKernelDim(func, dim3{}, dim3{}, args...);
    }

    template <typename Func, typename... Args>
    void runKernelLinear(Func func, const size_t size, Args ... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchLinear>(*mResourceManager,
                                                                         func, size, Impl::castID(args)...));
    }

    void addCallback(const std::function<void()>& func);
    void sync();

    void pushOperator(std::unique_ptr<Impl::Operator>&& op);
    void pushOperator(std::function<void(ID,ResourceManager&, Stream&)>&& op);

    template <typename T, typename... Args>
    auto makeLazyConstructor(Args ... args) {
        return Impl::LazyConstructor<T, decltype(Impl::castID(args))...>(Impl::castID(args)...);
    }

    ResourceManager& getResourceManager();
    std::unique_ptr<Task> bindStream(Stream& stream, std::shared_ptr<Impl::TaskState> promise);
};

class CommandBufferQueue final : Uncopyable {
public:
    using UnboundTask = std::pair<std::shared_ptr<Impl::TaskState>, std::unique_ptr<CommandBuffer>>;
private:
    std::queue<UnboundTask> mQueue;
    std::mutex mMutex;
public:
    void submit(std::shared_ptr<Impl::TaskState> promise, std::unique_ptr<CommandBuffer> buffer);
    UnboundTask getTask();
    size_t size() const;
    void clear();
};

using Clock = std::chrono::high_resolution_clock;

class DispatchSystem final : Uncopyable {
private:
    class StreamInfo final : Uncopyable {
    private:
        Stream mStream;
        std::unique_ptr<Task> mTask;
        std::vector<std::unique_ptr<Task>> mPool;
        Clock::time_point mLast;
    public:
        StreamInfo();
        bool free() const;
        void set(CommandBufferQueue::UnboundTask&& task);
        void update(Clock::time_point point);
        bool operator<(const StreamInfo& rhs) const;
    };

    StreamInfo& getStream();
    std::vector<StreamInfo> mStreams;
    CommandBufferQueue& mQueue;
public:
    explicit DispatchSystem(CommandBufferQueue& queue);
    void update();
};

template <typename T>
void Resource<T>::addInstance(std::unique_ptr<ResourceInstance> instance) const {
    mManager.registerResource(mID, std::move(instance));
}

template <typename T>
Resource<T>::Resource(ResourceManager& manager)
    : mID(manager.allocResource()), mManager(manager) {}
