#pragma once
#include <Base/Pipeline.hpp>
#include <map>
#include <queue>
#include <type_traits>
#include <chrono>

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
        virtual void set(int mask, Stream& stream);
    };

    class GlobalMemory final :public DeviceMemoryInstance {
    private:
        SharedMemory mMemory;
    public:
        GlobalMemory(size_t size, ID end);
        void* get() override;
        void set(const void* src, Stream& stream) override;
        void set(int mask, Stream& stream) override;
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

template<typename T>
class MemoryRef final:public Impl::DMRef {
public:
    MemoryRef(const std::shared_ptr<Impl::DeviceMemory>& ref) :DMRef(ref) {}
    size_t size() const {
        return Impl::DMRef::size() / sizeof(T);
    }
};

namespace Impl {

    template<typename T,typename=std::enable_if_t<!std::is_base_of<Impl::DMRef,T>::value>>
    T castID(T arg) {
        return arg;
    }

    struct Tag {};

    template<typename T>
    class TID final:public Tag {
    private:
        ID mID;
    public:
        TID(ID id):mID(id){}
        T* get(CommandBuffer& buffer) {
            return buffer.mDeviceMemory.find(mID)->second->get();
        }
    };

    template<typename T>
    auto castID(const MemoryRef<T>& ref)->TID<T> {
        return ref.getID();
    }

    template<typename T, typename = std::enable_if_t<!std::is_base_of<Tag, T>::value>>
    T castPtr(T arg) {
        return arg;
    }
    template<typename T>
    auto castPtr(TID<T> ref) {
        return ref.get();
    }

    class KernelLaunchDim final :public Operator {
    private:
        std::function<void(Stream&)> mClosure;
    public:
        template<typename Func, typename... Args>
        KernelLaunchDim(CommandBuffer& buffer, Func func, dim3 grid, dim3 block, Args... args)
            :Operator(buffer) {
            mClosure = [=](Stream& stream) {
                stream.runDim(func, grid, block, castPtr(args)...);
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

    class Flag final :public Operator {
    private:
        std::shared_ptr<bool> mPtr;
    public:
        Flag(CommandBuffer& buffer, const std::shared_ptr<bool>& ptr);
        void emit(Stream& stream) override;
    };
}

class Future final {
private:
    std::shared_ptr<bool> mPromise;
public:
    Future(const std::shared_ptr<bool>& promise);
    bool ready() const;
};

class CommandBuffer final :Uncopyable{
private:
    friend class Impl::DeviceMemory;
    friend class Impl::Operator;
    template<typename T>
    friend class Impl::TID;
    friend class DispatchSystem;
    void newDMI(ID id, size_t size, ID end,Impl::MemoryType type);
    std::map<ID, std::unique_ptr<Impl::DeviceMemoryInstance>> mDeviceMemory;
    std::queue<std::unique_ptr<Impl::Operator>> mCommandQueue;
    ID mLast;
    std::shared_ptr<bool> mPromise;
    void setPromise(const std::shared_ptr<bool>& promise);
public:
    CommandBuffer();
    template<typename T>
    MemoryRef<T> allocBuffer(size_t size) {
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
    void update(Stream& stream);
    bool ready() const;
};

class DispatchSystem final:Uncopyable {
private:
    std::vector<std::pair<size_t,std::unique_ptr<CommandBuffer>>> mTasks;
    std::vector<Stream> mStreams;
    size_t mPos,mAlloc;
public:
    DispatchSystem(size_t size);
    Future submit(std::unique_ptr<CommandBuffer>&& buffer);
    size_t size() const;
    void update(std::chrono::nanoseconds tot);
};
