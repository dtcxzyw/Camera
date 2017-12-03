#pragma once
#include <Base/Pipeline.hpp>
#include <map>

class CommandBuffer;

namespace Impl {
    /*
    struct AnyBase {
        virtual std::unique_ptr<AnyBase> clone() = 0;
        virtual ~AnyBase() = default;
    };
    template<typename T>
    struct AnyHolder final :AnyBase {
    private:
        T mData;
    public:
        AnyHolder(const T& rhs) :mData(rhs) {}
        AnyHolder(T&& rhs) :mData(rhs) {}
        std::unique_ptr<AnyHolder> clone() const override {
            return std::make_unique<AnyHolder>(mData);
        }
    };

    class Any final {
    private:
        std::unique_ptr<Impl::AnyBase> mData;
    public:
        template<typename T>
        Any(const T& rhs) :mData(std::make_unique<AnyHolder<T>>(rhs)) {}
        template<typename T>
        Any(T&& rhs) : mData(std::make_unique<AnyHolder<T>>(rhs)) {}
        Any(Any& rhs);
        bool vaild() const;
        void release();
        template<typename T>
        T& get() {
            return *reinterpret_cast<T*>(mData.get());
        }
        void swap(Any& rhs);
        template<typename T>
        Any& operator=(T&& rhs) {
            Any(rhs).swap(*this);
            return *this;
        }
        template<typename T>
        Any& operator=(const T& rhs) {
            Any(rhs).swap(*this);
            return *this;
        }
        template<typename T, typename... Args>
        void emplace(Args&&... args) {
            auto data = std::make_unique<AnyHolder<T>>(std::forward<Args>(args)...);
            mData.swap(data);
        }
    };
    */

    size_t getPID();

    enum MemoryType {
        global, constant
    };

    class DeviceMemory final :Uncopyable {
    private:
        size_t mID, mSize;
        CommandBuffer& mBuffer;
        MemoryType mType;
    public:
        DeviceMemory(CommandBuffer& buffer, size_t size,MemoryType type);
        ~DeviceMemory();
        size_t getID() const;
        size_t size() const;
    };

    class DeviceMemoryInstance:Uncopyable {
    private:
        size_t mEnd;
    protected:
        size_t mSize;
    public:
        DeviceMemoryInstance(size_t size, size_t end);
        virtual ~DeviceMemoryInstance()=default;
        bool shouldRelease(size_t current) const;
        virtual void* get()=0;
        virtual void set(const void* src,Stream& stream)=0;
        virtual void set(int mask, Stream& stream);
    };

    class GlobalMemory final:public DeviceMemoryInstance {
    private:
        SharedMemory mMemory;
    public:
        GlobalMemory(size_t size, size_t end);
        void* get() override;
        void set(const void* src, Stream& stream) override;
        void set(int mask, Stream& stream) override;
    };

    class ConstantMemory final :public DeviceMemoryInstance {
    private:
        void* mPtr;
    public:
        ConstantMemory(size_t size, size_t end);
        ~ConstantMemory();
        void* get() override;
        void set(const void* src, Stream& stream) override;
    };
}

template<typename T>
class DMRef final {
private:
    std::shared_ptr<Impl::DeviceMemory> mRef;
public:
    DMRef(const std::shared_ptr<Impl::DeviceMemory>& ref) :mRef(ref){}
    size_t getID() const {
        return mRef->getID();
    }
    size_t size() const{
        return mRef->size()/sizeof(T);
    }
};

class CommandBuffer final :Uncopyable{
private:
    friend class Impl::DeviceMemory;
    std::map<size_t,std::unique_ptr<Impl::DeviceMemoryInstance>> mDeviceMemory;
    void newDMI(size_t id, size_t size, size_t end,Impl::MemoryType type);
public:
    template<typename T>
    DMRef<T> allocBuffer(size_t size) {
        return std::make_shared<Impl::DeviceMemory>(*this, size*sizeof(T),Impl::MemoryType::global);
    }
    template<typename T>
    DMRef<T> allocConstant() {
        return std::make_shared<Impl::DeviceMemory>(*this,sizeof(T),Impl::MemoryType::constant);
    }

};

class DispatchSystem final:Uncopyable {
private:

public:

};

