#pragma once
#include <Core/Pipeline.hpp>
#include <map>
#include <queue>
#include <type_traits>
#include <functional>
#include <limits>
#ifdef CAMERA_RESOURCE_CHECK
#include <set>
#endif

class CommandBuffer;
class ResourceManager;
class StreamContext;
class Task;

namespace Impl {
    struct TaskState;
}

using Id = uint32_t;

class ResourceInstance : Uncopyable {
public:
    ResourceInstance() = default;
    virtual ~ResourceInstance() = default;
    virtual bool canBeRecycled() const;
    virtual void bindStream(StreamContext&);
    virtual void getRes(void*, cudaStream_t) = 0;
};

template <typename T>
class Resource : Uncopyable {
private:
    const Id mId;
protected:
    ResourceManager& mManager;
    void addInstance(std::unique_ptr<ResourceInstance> instance) const;
public:
    explicit Resource(ResourceManager& manager);
    virtual ~Resource() = default;

    Id getId() const noexcept {
        return mId;
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

    Id getId() const {
        return mRef->getId();
    }
};

using MemoryReleaseFunction = std::function<void(UniqueMemory, size_t)>;

template <typename Func>
struct KernelDesc final {
    Func func;
    size_t stackSize;
    KernelDesc(const Func func, const size_t stackSize) : func(func), stackSize(stackSize) {}
};

template <typename Func>
auto makeKernelDesc(const Func func, const size_t stackSize = 1024U) {
    return KernelDesc<Func>(func, stackSize);
}

namespace Impl {
    class DeviceMemoryDesc : public Resource<void*> {
    protected:
        const size_t mSize;
    public:
        DeviceMemoryDesc(ResourceManager& manager, size_t size);
        size_t size() const;
    };

    class GlobalMemoryDesc final : public DeviceMemoryDesc {
    private:
        MemoryReleaseFunction mOnRelease;
    public:
        GlobalMemoryDesc(ResourceManager& manager, size_t size,
            MemoryReleaseFunction onRelease);
        virtual ~GlobalMemoryDesc();
    };

    class ConstantMemoryDesc final : public DeviceMemoryDesc {
    public:
        ConstantMemoryDesc(ResourceManager& manager, size_t size);
        virtual ~ConstantMemoryDesc();
    };

    class AllocatedMemoryDesc final : public DeviceMemoryDesc {
    private:
        MemorySpan<unsigned char> mRef;
    public:
        AllocatedMemoryDesc(ResourceManager& manager,
            const MemorySpan<unsigned char>& ref);
        virtual ~AllocatedMemoryDesc();
    };

    class DeviceMemoryInstance : public ResourceInstance {
    protected:
        const size_t mSize;
        void getRes(void*, cudaStream_t) override;
    public:
        explicit DeviceMemoryInstance(size_t size);
        virtual ~DeviceMemoryInstance() = default;
        virtual void* get() = 0;
        virtual void set(const void* src, size_t begin, size_t end, Stream& stream) = 0;
        virtual void memset(int mask, size_t begin, size_t end, Stream& stream);
    };

    class L1GlobalMemoryPool;

    class GlobalMemory final : public DeviceMemoryInstance {
    private:
        UniqueMemory mMemory;
        L1GlobalMemoryPool* mPool;
        MemoryReleaseFunction mOnRelease;
        bool canBeRecycled() const override;
    public:
        GlobalMemory(size_t size, MemoryReleaseFunction onRelease);
        void bindStream(StreamContext& info) override;
        ~GlobalMemory();
        void* get() override;
        void set(const void* src, size_t begin, size_t end, Stream& stream) override;
        void memset(int mask, size_t begin, size_t end, Stream& stream) override;
    };

    class ConstantMemory final : public DeviceMemoryInstance {
    private:
        void* mPtr;
    public:
        explicit ConstantMemory(size_t size);
        ~ConstantMemory();
        void* get() override;
        void set(const void* src, size_t begin, size_t end, Stream& stream) override;
    };

    class AllocatedMemory final : public DeviceMemoryInstance {
    private:
        MemorySpan<unsigned char> mRef;
    public:
        explicit AllocatedMemory(const MemorySpan<unsigned char>& ref);
        void* get() override;
        void set(const void* src, size_t begin, size_t end, Stream& stream) override;
        void memset(int mask, size_t begin, size_t end, Stream& stream) override;
    };

    class CastTag {
    protected:
        static void get(ResourceManager& manager, Id id, void* ptr);
    };

    template <typename T>
    class ResourceId final : public CastTag {
    private:
        Id mId;
    public:
        explicit ResourceId(const Id id) : mId(id) {}

        T get(ResourceManager& manager) {
            T res;
            CastTag::get(manager, mId, &res);
            return res;
        }
    };

    template <typename T>
    class SpanHelper final : public CastTag {
    private:
        Id mId;
        size_t mBegin, mEnd;
    public:
        SpanHelper() : mId(0), mBegin(0), mEnd(0) {}

        SpanHelper(const Id id, const size_t begin, const size_t end)
            : mId(id), mBegin(begin), mEnd(end) {}

        T* get(ResourceManager& manager) {
            if (mId) {
                T* res;
                CastTag::get(manager, mId, &res);
                return res + mBegin;
            }
            return nullptr;
        }

        size_t begin() const {
            return mBegin;
        }

        size_t end() const {
            return mEnd;
        }

        Id getId() const {
            return mId;
        }
    };

    template <typename T>
    class Span final {
    private:
        std::shared_ptr<DeviceMemoryDesc> mRef;
        size_t mBegin, mEnd;
    public:
        template <typename U>
        friend class Span;

        Span() : mBegin(0), mEnd(0) {}

        explicit Span(const std::shared_ptr<DeviceMemoryDesc>& ref)
            : mRef(ref), mBegin(0), mEnd(mRef->size() / sizeof(T)) {}

        template <typename U>
        Span(const Span<U>& rhs) : mRef(rhs.mRef), mBegin(rhs.mBegin * sizeof(U) / sizeof(T)),
            mEnd(rhs.mEnd * sizeof(U) / sizeof(T)) {
            #ifdef CAMERA_DEBUG
            if (rhs.mBegin % sizeof(T) != 0 || rhs.mEnd % sizeof(T) != 0)
                throw std::logic_error("bad cast");
            #endif
        }

        Span subSpan(const size_t begin, const size_t end = std::numeric_limits<size_t>::max()) const {
            #ifdef CAMERA_DEBUG
            if (mBegin + begin > mEnd)throw std::logic_error("bad cast");
            #endif
            Span res;
            res.mRef = mRef;
            res.mBegin = mBegin + begin;
            if (end == std::numeric_limits<size_t>::max())res.mEnd = mEnd;
            else res.mEnd = mBegin + end;
            return res;
        }

        size_t size() const {
            return mEnd - mBegin;
        }

        size_t maxSize() const {
            return calcMaxBufferSize<T>(size());
        }

        operator SpanHelper<T>() const {
            if (mRef)return {mRef->getId(), mBegin, mEnd};
            return {};
        }

        void reset() {
            mRef.reset();
            mBegin = mEnd = 0;
        }
    };

    template <typename T, typename = std::enable_if_t<!std::is_base_of<ResourceTag, T>::value>>
    T castId(T arg) {
        return arg;
    }

    template <typename T>
    auto castId(const ResourceRef<T>& ref) -> ResourceId<T> {
        return ResourceId<T>{ref.getId()};
    }

    template <typename T>
    auto castId(const Span<T>& ref) {
        return SpanHelper<T>(ref);
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
        auto constructImpl(ResourceManager& manager, std::index_sequence<I...>) const {
            return T{cast(std::get<I>(mArgs), manager)...};
        }

    public:
        explicit LazyConstructor(Args ... args) : mArgs(std::make_tuple(args...)) {}

        auto get(ResourceManager& manager) const {
            return constructImpl(manager, std::make_index_sequence<std::tuple_size<decltype(mArgs)>::value>());
        }
    };

    class LaunchSize final {
    private:
        SpanHelper<uint32_t> mHelper;
        Span<uint32_t> mRef;
    public:
        explicit LaunchSize(const Span<uint32_t>& ptr);
        SpanHelper<uint32_t> get() const;
        void download(uint32_t& dst, CommandBuffer& buffer) const;
    };

    template <typename T>
    class ValueHelper final : public CastTag {
    private:
        std::function<T(ResourceManager&)> mClosure;
    public:
        template <typename U>
        explicit ValueHelper(const U& val) : mClosure(
            [rval = castId(val)](ResourceManager& manager) {
                return cast(rval, manager);
            }) {}

        T get(ResourceManager& manager) {
            return mClosure(manager);
        }
    };

    template <typename T>
    class Value final {
    private:
        std::function<ValueHelper<T>()> mHolder;
    public:
        template <typename U>
        Value(U val) : mHolder([val] {
            return ValueHelper<T>(val);
        }) {}

        auto get() const {
            return mHolder();
        }
    };

    class Operator : Uncopyable {
    protected:
        ResourceManager& mManager;
        Id mId;
        DeviceMemoryInstance& getMemory(Id id) const;
    public:
        explicit Operator(ResourceManager& manager);
        virtual ~Operator() = default;
        Id getId() const;
        virtual void emit(Stream& stream) = 0;
    };

    class FunctionOperator final : public Operator {
    private:
        std::function<void(Id, ResourceManager&, Stream&)> mClosure;
    public:
        FunctionOperator(ResourceManager& manager,
            std::function<void(Id, ResourceManager&, Stream&)>&& closure);
        void emit(Stream& stream) override;
    };

    class Memset final : public Operator {
    private:
        SpanHelper<unsigned char> mSpan;
        int mMask;
    public:
        Memset(ResourceManager& manager, SpanHelper<unsigned char> span, int mask);
        void emit(Stream& stream) override;
    };

    class Memcpy final : public Operator {
    private:
        SpanHelper<unsigned char> mDst;
        std::function<void(std::function<void(const void*)>)> mSrc;
    public:
        Memcpy(ResourceManager& manager, SpanHelper<unsigned char> dst,
            std::function<void(std::function<void(const void*)>)>&& src);
        void emit(Stream& stream) override;
    };

    void setCallStackSize(size_t size);

    class KernelLaunchDim final : public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template <typename Func, typename... Args>
        KernelLaunchDim(ResourceManager& manager, const KernelDesc<Func> func, const dim3 grid,
            const dim3 block, Args ... args) : Operator(manager) {
            mClosure = [=, &manager](Stream& stream) {
                setCallStackSize(func.stackSize);
                stream.launchDim(func.func, grid, block, cast(args, manager)...);
            };
        }

        void emit(Stream& stream) override;
    };

    class KernelLaunchLinear final : public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template <typename Func, typename... Args>
        KernelLaunchLinear(ResourceManager& manager, const KernelDesc<Func> func,
            const size_t size, Args ... args) : Operator(manager) {
            mClosure = [=, &manager](Stream& stream) {
                setCallStackSize(func.stackSize);
                stream.launchLinear(func.func, size, cast(args, manager)...);
            };
        }

        void emit(Stream& stream) override;
    };

}

using Impl::LaunchSize;
using Impl::Span;
using Impl::Value;

class ResourceRecycler : Uncopyable {
protected:
    uint64_t mCurrent;
public:
    ResourceRecycler();
    virtual ~ResourceRecycler() = default;
    virtual void gc(uint64_t id);
    void setCurrent(uint64_t id);
};

class ResourceManager final : Uncopyable {
private:
    std::map<Id, std::pair<Id, std::unique_ptr<ResourceInstance>>> mResources;
    cudaStream_t mStream = nullptr;
    Id mResourceCount = 0, mOperatorCount = 0, mSyncPoint = 0;
    #ifdef CAMERA_RESOURCE_CHECK
    std::set<Id> mUnknownResource;
    #endif
public:
    void registerResource(Id id, std::unique_ptr<ResourceInstance>&& instance);
    ResourceInstance& getResource(Id id);
    void bindStream(StreamContext& stream);
    cudaStream_t getStream() const;
    void gc(Id time);
    Id allocResource();
    Id getOperatorPid();
    void syncPoint(Id time);
};

class CommandBuffer final : Uncopyable {
private:
    std::unique_ptr<ResourceManager> mResourceManager;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
public:
    CommandBuffer();

    template <typename T>
    Span<T> allocBuffer(const size_t size = 1,
        MemoryReleaseFunction onRelease = {}) {
        return Span<T>{
            std::make_shared<Impl::GlobalMemoryDesc>(*mResourceManager, size * sizeof(T),
                std::move(onRelease))
        };
    }

    template <typename T>
    Span<T> allocConstant(const size_t size = 1) {
        return Span<T>{
            std::make_shared<Impl::ConstantMemoryDesc>(*mResourceManager, size * sizeof(T))
        };
    }

    template <typename T>
    Span<T> useAllocated(const MemorySpan<T>& ref) {
        return Span<T>{
            std::make_shared<Impl::AllocatedMemoryDesc>(*mResourceManager,
                static_cast<MemorySpan<unsigned char>>(ref))
        };
    }

    void memset(const Span<unsigned char>& memory, int mask = 0);

    void memcpy(const Span<unsigned char>& dst,
        std::function<void(std::function<void(const void*)>)>&& src);

    void memcpy(const Span<unsigned char>& dst, const MemorySpan<unsigned char>& src);

    template <typename T>
    void memcpy(Span<T>& dst, const Span<T>& src) {
        const auto id = src.getID();
        memcpy(dst, [id, this](auto call) {
            void* res;
            mResourceManager->getResource(id).getRes(&res, mResourceManager->getStream());
            call(res);
        });
    }

    template <typename Func, typename... Args>
    void launchKernelDim(const KernelDesc<Func> func, const dim3 grid, const dim3 block, Args ... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchDim>(*mResourceManager,
            func, grid, block, Impl::castId(args)...));
    }

    template <typename Func, typename... Args>
    void callKernel(const KernelDesc<Func> func, Args ... args) {
        launchKernelDim(func, dim3{}, dim3{}, args...);
    }

    template <typename Func, typename... Args>
    void launchKernelLinear(const KernelDesc<Func> func, const size_t size, Args ... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchLinear>(*mResourceManager,
            func, size, Impl::castId(args)...));
    }

    void addCallback(const std::function<void()>& func);
    void sync();

    void pushOperator(std::unique_ptr<Impl::Operator>&& op);
    void pushOperator(std::function<void(Id, ResourceManager&, Stream&)>&& op);

    template <typename T, typename... Args>
    auto makeLazyConstructor(Args ... args) {
        return Impl::LazyConstructor<T, decltype(Impl::castId(args))...>(Impl::castId(args)...);
    }

    ResourceManager& getResourceManager();
    std::unique_ptr<Task> bindStream(StreamContext& stream, std::shared_ptr<Impl::TaskState> promise);
};

template <typename T>
void Resource<T>::addInstance(std::unique_ptr<ResourceInstance> instance) const {
    mManager.registerResource(mId, std::move(instance));
}

template <typename T>
Resource<T>::Resource(ResourceManager& manager)
    : mId(manager.allocResource()), mManager(manager) {}
