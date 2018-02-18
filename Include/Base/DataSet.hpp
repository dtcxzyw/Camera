#pragma once
#include <Base/Common.hpp>

template<typename Enum, Enum Name, typename T>
struct UnitInfo {
    static constexpr auto name = Name;
    using Type = T;
};

#define VAR(name,type) UnitInfo<decltype(name),name,type>

namespace Impl {

    template<typename Enum, Enum Name>
    struct Tag final {};

    template<typename First, typename... Others>
    class DataSet :DataSet<Others...> {
    private:
        using T = typename First::Type;
        using Enum = decltype(First::name);
        T mData;
    protected:
        CUDA auto& getImpl(Tag<Enum, First::name>) {
            return mData;
        }
        template<Enum name>
        CUDA auto& getImpl(Tag<Enum, name>) {
            return DataSet<Others...>::getImpl(Tag<Enum, name>{});
        }
        CUDA const auto& getImpl(Tag<Enum, First::name>) const{
            return mData;
        }
        template<Enum name>
        CUDA const auto& getImpl(Tag<Enum, name>) const{
            return DataSet<Others...>::getImpl(Tag<Enum, name>{});
        }
    public:
        CUDA DataSet() {};
        CUDA DataSet(T first,DataSet<Others...> others):DataSet<Others...>(others),mData(first){}
        template<Enum Name>
        CUDA auto& get() {
            return getImpl(Tag<Enum, Name>{});
        }
        template<Enum Name>
        CUDA auto get() const{
            return getImpl(Tag<Enum, Name>{});
        }
        CUDA DataSet operator*(float rhs) const {
            return DataSet{ static_cast<T>(mData*rhs),DataSet<Others...>::operator*(rhs) };
        }
        CUDA DataSet operator+(DataSet rhs) const {
            return DataSet{ static_cast<T>(mData+rhs.mData),DataSet<Others...>::operator+(rhs) };
        }
    };

    template<>
    class DataSet<void> {
    public:
        CUDA DataSet operator*(float) const {
            return *this;
        }
        CUDA DataSet operator+(DataSet) const {
            return *this;
        }
    };

}

template<typename... Units>
using Args = Impl::DataSet<Units..., void>;

using EmptyArg = Args<>;
struct Empty final{};
