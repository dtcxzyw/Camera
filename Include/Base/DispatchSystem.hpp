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
using ID = uint64_t;

namespace Impl {
    ID getPID();
    struct ResourceTag {};
}

class ResourceInstance :Uncopyable {
private:
    ID mEnd;
public:
    ResourceInstance();
    virtual ~ResourceInstance() = default;
    bool shouldRelease(ID current) const;
    virtual void getRes(void*, cudaStream_t) = 0;
};

template<typename T>
class Resource :Uncopyable {
private:
    ID mID;
    CommandBuffer& mBuffer;
protected:
    void addInstance(std::unique_ptr<ResourceInstance>&& instance) const;
public:
    explicit Resource(CommandBuffer& buffer) :mID(Impl::getPID()), mBuffer(buffer) {}
    virtual ~Resource() = default;
    ID getID() const noexcept{
        return mID;
    }
};

template<typename T>
class ResourceRef :Impl::ResourceTag {
protected:
    std::shared_ptr<Resource<T>> mRef;
public:
    explicit ResourceRef(std::shared_ptr<Resource<T>>  ref) :mRef(std::move(ref)) {}
    void earlyRelease() {
        mRef.reset();
    }
    ID getID() const {
        return mRef->getID();
    }
};

namespace Impl {
    enum class MemoryType {
        Global, Constant
    };

    class DeviceMemory final :public Resource<void*> {
    private:
        size_t mSize;
        MemoryType mType;
    public:
        DeviceMemory(CommandBuffer& buffer, size_t size, MemoryType type);
        virtual ~DeviceMemory();
        size_t size() const;
    };

    class DeviceMemoryInstance :public ResourceInstance {
    protected:
        size_t mSize;
        void getRes(void*, cudaStream_t) override;
    public:
        explicit DeviceMemoryInstance(size_t size);
        virtual ~DeviceMemoryInstance() = default;
        virtual void* get() = 0;
        virtual void set(const void* src, Stream& stream) = 0;
        virtual void memset(int mask, Stream& stream);
    };

    class GlobalMemory final :public DeviceMemoryInstance {
    private:
        SharedMemory mMemory;
    public:
        explicit GlobalMemory(size_t size);
        void* get() override;
        void set(const void* src, Stream& stream) override;
        void memset(int mask, Stream& stream) override;
    };

    class ConstantMemory final :public DeviceMemoryInstance {
    private:
        void* mPtr;
    public:
        explicit ConstantMemory(size_t size);
        ~ConstantMemory();
        void* get() override;
        void set(const void* src, Stream& stream) override;
    };

    class DMRef :public ResourceRef<void*> {
    public:
        explicit DMRef(const std::shared_ptr<DeviceMemory>& ref);
        size_t size() const;
    };

    class Operator :Uncopyable {
    protected:
        CommandBuffer & mBuffer;
        ID mID;
        DeviceMemoryInstance& getMemory(ID id) const;
    public:
        explicit Operator(CommandBuffer& buffer);
        virtual ~Operator() = default;
        ID getID() const;
        virtual void emit(Stream& stream) = 0;
    };

    class Memset final :public Operator {
    private:
        ID mMemoryID;
        int mMask;
    public:
        Memset(CommandBuffer& buffer, ID memoryID, int mask);
        void emit(Stream& stream) override;
    };

    class Memcpy final :public Operator {
    private:
        ID mDst;
        std::function<void(std::function<void(const void*)>)> mSrc;
    public:
        Memcpy(CommandBuffer& buffer, ID dst
            , std::function<void(std::function<void(const void*)>)>&& src);
        void emit(Stream& stream) override;
    };
}

class FunctionOperator final :public Impl::Operator {
private:
    std::function<void(Stream&)> mClosure;
public:
    FunctionOperator(CommandBuffer& buffer, std::function<void(Stream&)>&& closure);
    void emit(Stream& stream) override;
};

template<typename T>
class MemoryRef final :public Impl::DMRef {
public:
    explicit MemoryRef(const std::shared_ptr<Impl::DeviceMemory>& ref) :DMRef(ref) {}
    size_t size() const {
        return DMRef::size() / sizeof(T);
    }
};

namespace Impl {
    class CastTag {
    protected:
        static void get(CommandBuffer & buffer, ID id, void* ptr);
    };

    template<typename T>
    class ResourceID final :public CastTag {
    private:
        ID mID;
    public:
        explicit ResourceID(const ID id) :mID(id) {}
        T get(CommandBuffer& buffer) {
            T res;
            CastTag::get(buffer, mID, &res);
            return res;
        }
    };

    template<typename T, typename = std::enable_if_t<!std::is_base_of<Impl::ResourceTag, T>::value>>
    T castID(T arg) {
        return arg;
    }

    template<typename T>
    auto castID(const ResourceRef<T>& ref)->ResourceID<T> {
        return ResourceID<T>{ref.getID()};
    }

    template<typename T>
    auto castID(const MemoryRef<T>& ref)->ResourceID<T*> {
        return ResourceID<T*>{ref.getID()};
    }

    template<typename T, typename = std::enable_if_t<!std::is_base_of<CastTag, T>::value>>
    T cast(T arg, CommandBuffer&) {
        return arg;
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of<CastTag, T>::value>>
    auto cast(T ref, CommandBuffer& buffer) {
        return ref.get(buffer);
    }

    template<typename T, typename... Args>
    class LazyConstructor final :public CastTag {
    private:
        std::tuple<Args...> mArgs;
        template<size_t... I>
        auto constructImpl(CommandBuffer& buffer, std::index_sequence<I...>) {
            return T{ cast(std::get<I>(mArgs),buffer)... };
        }
    public:
        explicit LazyConstructor(Args... args) :mArgs(std::make_tuple(args...)) {}
        auto get(CommandBuffer& buffer) {
            return constructImpl(buffer, std::make_index_sequence<std::tuple_size<decltype(mArgs)>::value>());
        }
    };

    template<typename T>
    class DataPtrHelper final :public CastTag {
    private:
        std::function<T*(CommandBuffer&)> mClosure;
    public:
        explicit DataPtrHelper(const MemoryRef<T>& ref) {
            auto rval = castID(ref);
            mClosure = [rval](CommandBuffer& buffer) {
                return cast(rval, buffer);
            };
        }

        explicit DataPtrHelper(const DataViewer<T>& data) {
            auto ptr = data.begin();
            mClosure = [ptr](CommandBuffer& buffer) {
                return ptr;
            };
        }
        T* get(CommandBuffer& buffer) {
            return mClosure(buffer);
        }
    };

    template<typename T>
    class DataPtr final {
    private:
        std::function<DataPtrHelper<T>()> mClosure;
        size_t mSize;
    public:
        DataPtr(MemoryRef<T> ref):mSize(ref.size()) {
            mClosure = [ref] {return DataPtrHelper<T>(ref); };
        }
        DataPtr(const DataViewer<T>& data):mSize(data.size()) {
            mClosure = [data] {return DataPtrHelper<T>(data);};
        }
        auto get() const {
            return mClosure();
        }
        auto size() const {
            return mSize;
        }
    };

    template<typename T>
    class ValueHelper final :public CastTag {
    private:
        std::function<T(CommandBuffer&)> mClosure;
    public:
        template<typename U>
        explicit ValueHelper(const U& val) {
            auto rval = castID(val);
            mClosure = [rval](CommandBuffer& buffer) {
                return cast(rval, buffer);
            };
        }
        T get(CommandBuffer& buffer) {
            return mClosure(buffer);
        }
    };

    template<typename T>
    class Value final {
    private:
        std::function<ValueHelper<T>()> mClosure;
    public:
        template<typename U>
        Value(U val) {
            mClosure = [val] {return ValueHelper<T>(val); };
        }
        auto get() const {
            return mClosure();
        }
    };

    struct LaunchSizeHelper final :public CastTag {
    private:
        std::function<unsigned int*(CommandBuffer&)> mClosure;
    public:
        LaunchSizeHelper(const MemoryRef<unsigned int>& ptr,unsigned int off) {
            auto rval = castID(ptr);
            mClosure = [rval,off](CommandBuffer& buffer) {
                return cast(rval, buffer)+off;
            };
        }
        unsigned int* get(CommandBuffer& buffer) const {
            return mClosure(buffer);
        }
    };

    struct LaunchSize final {
    private:
        LaunchSizeHelper mHelper;
        MemoryRef<unsigned int> mRef;
    public:
        explicit LaunchSize(const MemoryRef<unsigned int>& ptr, const unsigned int off = 0)
            :mHelper(ptr, off), mRef(ptr) {}
        auto get() const {
            return mHelper;
        }
        void download(std::atomic_uint & dst,CommandBuffer& buffer) const;
    };

    class KernelLaunchDim final :public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template<typename Func, typename... Args>
        KernelLaunchDim(CommandBuffer& buffer, Func func, const dim3 grid, const dim3 block,
            Args... args):Operator(buffer) {
            mClosure = [=, &buffer](Stream& stream) {
                stream.runDim(func, grid, block, cast(args, buffer)...);
            };
        }
        void emit(Stream& stream) override;
    };

    class KernelLaunchLinear final :public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template<typename Func, typename... Args>
        KernelLaunchLinear(CommandBuffer& buffer, Func func, const size_t size, Args... args)
            :Operator(buffer) {
            mClosure = [=, &buffer](Stream& stream) {
                stream.run(func, size, cast(args, buffer)...);
            };
        }
        void emit(Stream& stream) override;
    };

}

using Impl::DataPtr;
using Impl::Value;
using Impl::LaunchSize;

class Future final {
private:
    std::shared_ptr<bool> mPromise;
public:
    explicit Future(std::shared_ptr<bool> promise);
    bool finished() const;
};

namespace Impl {
    void CUDART_CB updateLast(cudaStream_t, cudaError_t, void *);
}

class CommandBuffer final :Uncopyable {
private:
    template<typename T>
    friend class Resource;
    friend class Impl::Operator;
    friend class Impl::CastTag;
    friend class DispatchSystem;
    friend void Impl::updateLast(cudaStream_t, cudaError_t, void *);
    std::map<ID, std::unique_ptr<ResourceInstance>> mResource;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
    ID mLast,mUpdated;
    std::shared_ptr<bool> mPromise;
    cudaStream_t mStream = nullptr;
    void registerResource(ID id, std::unique_ptr<ResourceInstance>&& instance);
    void setPromise(const std::shared_ptr<bool>& promise);
    void update(Stream& stream);
    bool finished() const;
    bool isDone() const;
    ResourceInstance& getResource(ID id);
    cudaStream_t getStream() const;
public:
    CommandBuffer();
    template<typename T>
    MemoryRef<T> allocBuffer(const size_t size = 1) {
        return MemoryRef<T>{std::make_shared<Impl::DeviceMemory>(*this, size * sizeof(T),
            Impl::MemoryType::Global)};
    }

    template<typename T>
    MemoryRef<T> allocConstant() {
        return MemoryRef<T>{std::make_shared<Impl::DeviceMemory>(*this, sizeof(T),
            Impl::MemoryType::Constant)};
    }

    void memset(Impl::DMRef& memory, int mark = 0);

    void memcpy(Impl::DMRef& dst, std::function<void(std::function<void(const void*)>)>&& src);

    template<typename T>
    void memcpy(MemoryRef<T>& dst, const DataViewer<T>& src) {
        memcpy(dst, [src](auto call) {call(src.begin()); });
    }

    template<typename T>
    void memcpy(MemoryRef<T>& dst, const MemoryRef<T>& src) {
        auto id = src.getID();
        memcpy(dst, [id, this](auto call) {
            void* res;
            mResource.find(id)->second->getRes(&res);
            call(res);
        });
    }

    template<typename Func, typename... Args>
    void runKernelDim(Func func,const dim3 grid,const dim3 block, Args... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchDim>(*this, func, grid, block
            , Impl::castID(args)...));
    }

    template<typename Func, typename... Args>
    void callKernel(Func func, Args... args) {
        runKernelDim(func, dim3{}, dim3{}, args...);
    }

    template<typename Func, typename... Args>
    void runKernelLinear(Func func,const size_t size, Args... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchLinear>(*this, func,
            size, Impl::castID(args)...));
    }

    void addCallback(cudaStreamCallback_t func, void* data);
    void sync();

    void pushOperator(std::unique_ptr<Impl::Operator>&& op);
    void pushOperator(std::function<void(Stream&)>&& op);

    template<typename T, typename... Args>
    auto makeLazyConstructor(Args... args) {
        return Impl::LazyConstructor <T, decltype(Impl::castID(args))...>(Impl::castID(args)...);
    }
};

using Clock = std::chrono::high_resolution_clock;

class DispatchSystem final :Uncopyable {
private:
    std::queue<std::unique_ptr<CommandBuffer>> mTasks;
    class StreamInfo final :Uncopyable {
    private:
        Stream mStream;
        std::unique_ptr<CommandBuffer> mTask;
        std::vector<std::unique_ptr<CommandBuffer>> mPool;
        Clock::time_point mLast;
    public:
        StreamInfo();
        bool free() const;
        void set(std::unique_ptr<CommandBuffer>&& task);
        void update(Clock::time_point point);
        bool operator<(const StreamInfo& rhs) const;
    };
    std::vector<StreamInfo> mStreams;
    void update(Clock::time_point t);
public:
    explicit DispatchSystem(size_t size);
    Future submit(std::unique_ptr<CommandBuffer>&& buffer);
    size_t size() const;
    void update(std::chrono::nanoseconds tot);
    void update();
};

template<typename T>
void Resource<T>::addInstance(std::unique_ptr<ResourceInstance>&& instance) const {
    mBuffer.registerResource(mID, std::move(instance));
}
