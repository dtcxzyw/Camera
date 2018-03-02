#pragma once
#include <Base/DispatchSystem.hpp>

template<typename Index,typename... Args>
class IndexDescriptor final {
private:
    Impl::LazyConstructor<Index,Args...> mLazyConstructor;
    size_t mSize;
public:
    using IndexType = Index;
    explicit IndexDescriptor(const size_t size, Args... args)
        :mLazyConstructor(args...),mSize(size) {}
    auto size() const{
        return mSize;
    }
    auto get() const {
        return mLazyConstructor;
    }
};

template<typename Index,typename... Args>
auto makeIndexDescriptor(const size_t size,Args... args) {
    return IndexDescriptor<Index, decltype(Impl::castID(args))...>(size, Impl::castID(args)...);
}
