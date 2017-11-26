#pragma once
#include <type_traits>
#include "Common.hpp"

template<typename Enum, Enum Name, typename T>
struct UnitInfo {
    static constexpr auto name = Name;
    using Type = T;
};

#define Var(name,type) UnitInfo<decltype(name),name,type>

namespace Impl {

    template<typename Enum, Enum tag>
    struct Tag final {};

    template<typename First, typename... Others>
    class DataSet :DataSet<Others...> {
    private:
        using T = typename First::Type;
        using Enum = decltype(First::name);
        T data;
    protected:
        CUDA auto& getImpl(Tag<Enum, First::name>) {
            return data;
        }
        template<Enum name>
        CUDA auto& getImpl(Tag<Enum, name>) {
            return DataSet<Others...>::getImpl(Tag<Enum, name>{});
        }
        CUDA const auto& getImpl(Tag<Enum, First::name>) const{
            return data;
        }
        template<Enum name>
        CUDA const auto& getImpl(Tag<Enum, name>) const{
            return DataSet<Others...>::getImpl(Tag<Enum, name>{});
        }
    public:
        CUDA DataSet() = default;
        CUDA DataSet(T first,DataSet<Others...> others):data(first),DataSet<Others...>(others){}
        template<Enum name>
        CUDA auto& get() {
            return getImpl(Tag<Enum, name>{});
        }
        template<Enum name>
        CUDA const auto& get() const{
            return getImpl(Tag<Enum, name>{});
        }
        CUDA DataSet operator*(float rhs) const {
            return DataSet{ static_cast<T>(data*rhs),DataSet<Others...>::operator*(rhs) };
        }
        CUDA DataSet operator+(const DataSet& rhs) const {
            return DataSet{ static_cast<T>(data+rhs.data),DataSet<Others...>::operator+(rhs) };
        }
    };

    template<>
    class DataSet<void> {
    public:
        CUDA DataSet operator*(float rhs) const {
            return *this;
        }
        CUDA DataSet operator+(const DataSet& rhs) const {
            return *this;
        }
    };

}

template<typename... Units>
using Args = Impl::DataSet<Units..., void>;

using EmptyArg = Args<>;
struct Empty final{};
