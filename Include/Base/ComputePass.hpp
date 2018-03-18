#pragma once
#include <Base/DispatchSystem.hpp>
#include <unordered_set>

class CachedValue:Uncopyable {
private:
    Id mId;
public:
    explicit CachedValue(Id id);
    Id getId() const;
    bool operator==(const CachedValue& val) const;
};

template<typename T>
class CachedValueHandle:Uncopyable {
private:
    
public:

};

struct CacheHasher final {
    using argument_type = CachedValue;
    using result_type = Id;
    Id operator()(const CachedValue& cache) const {
        return cache.getId();
    }
};

class ComputePass final :Uncopyable {
private:
    std::unordered_set<CachedValue,CacheHasher> mCachedValues;
public:

};
