#pragma once
#include <Base/Pipeline.hpp>
#include <map>
#include <queue>
#include <type_traits>
#include <chrono>
#include <tuple>

class CommandBuffer;
using ID = uintmax_t;

namespace Impl {

    ID getPID();

    enum MemoryType {
        global, constant
    };

    class DeviceMemory final :Uncopyable {
    private:
        ID mID;
        size_t mSize;
        CommandBuffer& mBuffer;
        MemoryType mType;
    public:
        DeviceMemory(CommandBuffer& buffer, size_t size, MemoryType type);
        ~DeviceMemory();
        ID getID() const;
        size_t size() const;
    };

    class DeviceMemoryInstance :Uncopyable {
    private:
        ID mEnd;
    protected:
        size_t mSize;
    public:
        DeviceMemoryInstance(size_t size, ID end);
        virtual ~DeviceMemoryInstance() = default;
        bool shouldRelease(ID current) const;
        virtual void* get() = 0;
        virtual void set(const void* src, Stream& stream) = 0;
        virtual void memset(int mask, Stream& stream);
    };

    class GlobalMemory final :public DeviceMemoryInstance {
    private:
        SharedMemory mMemory;
    public:
        GlobalMemory(size_t size, ID end);
        void* get() override;
        void set(const void* src, Stream& stream) override;
        void memset(int mask, Stream& stream) override;
    };

    class ConstantMemory final :public DeviceMemoryInstance {
    private:
        void* mPtr;
    public:
        ConstantMemory(size_t size, ID end);
        ~ConstantMemory();
        void* get() override;
        void set(const void* src, Stream& stream) override;
    };

    class DMRef {
    private:
        std::shared_ptr<Impl::DeviceMemory> mRef;
    public:
        DMRef(const std::shared_ptr<Impl::DeviceMemory>& ref) :mRef(ref) {}
        ID getID() const {
            return mRef->getID();
        }
        size_t size() const {
            return mRef->size();
        }
    };

    class Operator :Uncopyable {
    protected:
        CommandBuffer& mBuffer;
        ID mID;
        Impl::DeviceMemoryInstance& getMemory(ID memoryID);
    public:
        Operator(CommandBuffer& buffer);
        ID getID();
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
        std::function<void*()> mSrc;
    public:
        Memcpy(CommandBuffer& buffer, ID dst, std::function<void*()>&& src);
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
class MemoryRef final:public Impl::DMRef {
public:
    MemoryRef(const std::shared_ptr<Impl::DeviceMemory>& ref) :DMRef(ref) {}
    size_t size() const {
        return Impl::DMRef::size() / sizeof(T);
    }
};

namespace Impl {

    template<typename T, typename = std::enable_if_t<!std::is_base_of<Impl::DMRef, T>::value>>
    T castID(T arg) {
        return arg;
    }

    class CastTag {
    public:
        void* get(CommandBuffer& buffer, ID id);
    };

    template<typename T>
    class TID final :public CastTag {
    private:
        ID mID;
    public:
        TID(ID id) :mID(id) {}
        T* get(CommandBuffer& buffer) {
            return get(buffer, mID);
        }
    };

    template<typename T>
    auto castID(const MemoryRef<T>& ref)->TID<T> {
        return ref.getID();
    }

    template<typename T, typename = std::enable_if_t<!std::is_base_of<CastTag, T>::value>>
    T cast(T arg) {
        return arg;
    }

    template<typename T>
    auto cast(TID<T> ref) {
        return ref.get();
    }

    template<typename T, typename... Args>
    class LazyConstructor;

    template<typename T,typename... Args>
    auto cast(LazyConstructor<T,Args...> ref) {
        return ref.construct();
    }

    template<typename T, typename... Args>
    class LazyConstructor final:public CastTag {
    private:
        std::tuple<Args...> mArgs;
        template<size_t... I>
        auto constructImpl(std::index_sequence<I...>) {
            return T{ cast(std::get<I>(mArgs))... };
        }
    public:
        LazyConstructor(Args... args):mArgs(std::make_tuple(args...)){}
        auto construct() {
            return constructImpl(std::make_index_sequence<std::tuple_size<decltype(mArgs)>::value>());
        }
    };

    class KernelLaunchDim final :public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template<typename Func, typename... Args>
        KernelLaunchDim(CommandBuffer& buffer, Func func, dim3 grid, dim3 block, Args... args)
            :Operator(buffer) {
            mClosure = [=](Stream& stream) {
                stream.runDim(func, grid, block, cast(args)...);
            };
        }
        void emit(Stream& stream) override;
    };

    class KernelLaunchLinear final :public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template<typename Func, typename... Args>
        KernelLaunchLinear(CommandBuffer& buffer, Func func, size_t size, Args... args)
            :Operator(buffer) {
            mClosure = [=](Stream& stream) {
                stream.run(func,size, castPtr(args)...);
            };
        }
        void emit(Stream& stream) override;
    };
}

template<typename T,typename... Args>
auto makeLazyConstructor(Args... args) {
    return Impl::LazyConstructor <T, decltype(Impl::castID(args))...>(Impl::castID(args)...);
}

class Future final {
private:
    std::shared_ptr<bool> mPromise;
public:
    Future(const std::shared_ptr<bool>& promise);
    bool finished() const;
};

class CommandBuffer final :Uncopyable{
private:
    friend class Impl::DeviceMemory;
    friend class Impl::Operator;
    friend class Impl::CastTag;
    friend class DispatchSystem;
    void newDMI(ID id, size_t size, ID end,Impl::MemoryType type);
    std::map<ID, std::unique_ptr<Impl::DeviceMemoryInstance>> mDeviceMemory;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
    ID mLast;
    std::shared_ptr<bool> mPromise;
    void setPromise(const std::shared_ptr<bool>& promise);
    void update(Stream& stream);
    bool finished() const;
public:
    CommandBuffer();
    template<typename T>
    MemoryRef<T> allocBuffer(size_t size=1) {
        return std::make_shared<Impl::DeviceMemory>(*this, size*sizeof(T),Impl::MemoryType::global);
    }

    template<typename T>
    MemoryRef<T> allocConstant() {
        return std::make_shared<Impl::DeviceMemory>(*this,sizeof(T),Impl::MemoryType::constant);
    }

    void memset(Impl::DMRef& memory, int mark=0);

    void memcpy(Impl::DMRef& dst,std::function<void*()>&& src);

    template<typename T>
    void memcpy(MemoryRef<T>& dst,const DataViewer<T>& src) {
        memcpy(dst, [src] {return src.begin(); });
    }

    template<typename T>
    void memcpy(MemoryRef<T>& dst, const MemoryRef<T>& src) {
        memcpy(dst, [src,this] {return mDeviceMemory.find(src)->second->get(); });
    }

    template<typename Func,typename... Args>
    void runKernelDim(Func func,dim3 grid,dim3 block,Args... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchDim>(func,grid,block
            ,Impl::castID(args)...));
    }

    template<typename Func, typename... Args>
    void runKernelLinear(Func func, size_t size, Args... args) {
        mCommandQueue.emplace(std::make_unique<Impl::KernelLaunchLinear>(func,size
            , Impl::castID(args)...));
    }

    void pushOperator(std::unique_ptr<Impl::Operator>&& op);
    void pushOperator(std::function<void(Stream&)>&& op);
};

using Clock = std::chrono::high_resolution_clock;

class DispatchSystem final:Uncopyable {
private:
    std::queue<std::unique_ptr<CommandBuffer>> mTasks;
    class StreamInfo final:Uncopyable {
    private:
        Stream mStream;
        std::unique_ptr<CommandBuffer> mTask;
        std::vector<std::unique_ptr<CommandBuffer>> mPool;
        Clock::time_point mLast;
    public:
        StreamInfo();
        bool free();
        void set(std::unique_ptr<CommandBuffer>&& task);
        void update(Clock::time_point point);
        bool operator<(const StreamInfo& rhs) const;
    };
    std::vector<StreamInfo> mStreams;
public:
    DispatchSystem(size_t size);
    Future submit(std::unique_ptr<CommandBuffer>&& buffer);
    size_t size() const;
    void update(std::chrono::nanoseconds tot);
};
