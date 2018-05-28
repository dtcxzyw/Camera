#pragma once
#include <Core/CommandBuffer.hpp>
#include <chrono>
#include <cstring>

struct JudgeCore {};

class EmptyJudge final : public JudgeCore {
public:
    bool judge(const EmptyJudge&) const {
        return false;
    }
};

class VersionComparer final : public JudgeCore {
private:
    Id mCount;
public:
    VersionComparer() = default;
    explicit VersionComparer(const Id count) : mCount(count) {}

    bool judge(const VersionComparer& rhs) const {
        return mCount == rhs.mCount;
    }
};

class VersionCounter final {
private:
    Id mCount;
public:
    VersionCounter() : mCount(0) {}

    void count() {
        ++mCount;
    }

    auto get() const {
        return VersionComparer{mCount};
    }
};

template <typename Type>
class EqualComparer final : public JudgeCore {
private:
    Type mValue;
public:
    EqualComparer() = default;
    explicit EqualComparer(const Type& val) : mValue(val) {}

    const Type& get() const {
        return mValue;
    }

    bool judge(const EqualComparer& rhs) const {
        return mValue == rhs.mValue;
    }
};

template <typename Type>
auto makeEqualComparer(const Type& val) {
    return EqualComparer<Type>{val};
}

template <typename Type>
class BinaryComparer final : public JudgeCore {
private:
    unsigned char mData[sizeof(Type)];
public:
    BinaryComparer() = default;

    explicit BinaryComparer(const Type& val) {
        memcpy(mData, &val, sizeof(Type));
    }

    const Type& get() const {
        return *reinterpret_cast<const Type*>(mData);
    }

    bool judge(const BinaryComparer& rhs) const {
        return std::memcmp(mData, rhs.mData, sizeof(Type)) == 0;
    }
};

class TimeOutJudge final : public JudgeCore {
private:
    Clock::time_point mTimeStamp;
    Clock::duration mTimeOut;
public:
    TimeOutJudge() = default;

    explicit TimeOutJudge(const Clock::duration timeOut)
        : mTimeStamp(Clock::now()), mTimeOut(timeOut) {}

    bool judge(const TimeOutJudge& rhs) const {
        return rhs.mTimeStamp - mTimeStamp <= mTimeOut;
    }
};

template <typename Judge>
class Not final : public JudgeCore {
private:
    Judge mJudge;
public:
    Not() = default;
    explicit Not(const Judge& judge) : mJudge(judge) {}

    bool judge(const Not& rhs) const {
        return !mJudge.judge(rhs);
    }
};

template <typename Judge, typename = std::enable_if_t<std::is_base_of_v<JudgeCore, Judge>>>
auto operator!(const Judge& val) {
    return Not<Judge>(val);
}

template <typename L, typename R>
class And final : public JudgeCore {
private:
    L mLhs;
    R mRhs;
public:
    And() = default;
    And(const L& lhs, const R& rhs) : mLhs(lhs), mRhs(rhs) {}

    bool judge(const And& rhs) const {
        return mLhs.judge(rhs.mLhs) && mRhs.judge(rhs.mRhs);
    }
};

template <typename L, typename R, typename =
    std::enable_if_t<std::is_base_of_v<JudgeCore, L> && std::is_base_of_v<JudgeCore, R>>>
auto operator&&(const L& lhs, const R& rhs) {
    return And<L, R>(lhs, rhs);
}

template <typename L, typename R>
class Or final : public JudgeCore {
private:
    L mLhs;
    R mRhs;
public:
    Or() = default;
    Or(const L& lhs, const R& rhs) : mLhs(lhs), mRhs(rhs) {}

    bool judge(const Or& rhs) const {
        return mLhs.judge(rhs.mLhs) || mRhs.judge(rhs.mRhs);
    }
};

template <typename L, typename R, typename =
    std::enable_if_t<std::is_base_of_v<JudgeCore, L> && std::is_base_of_v<JudgeCore, R>>>
auto operator||(const L& lhs, const R& rhs) {
    return Or<L, R>(lhs, rhs);
}

template <typename Type, typename Judge>
class CachedValueHolder final : Uncopyable {
private:
    Type mValue;
    Judge mJudge;
public:
    CachedValueHolder(const Type& type, const Judge& judge) : mValue(type), mJudge(judge) {}

    bool vaild(const Judge& rhs) {
        return mJudge.judge(rhs);
    }

    Type& get() {
        return mValue;
    }
};

//Helper

template <typename Type, typename Judge>
using SharedCacheHolder = std::shared_ptr<CachedValueHolder<Type, Judge>>;

template <typename Type, typename Judge>
using CachedMemoryHolder = SharedCacheHolder<MemorySpan<Type>, Judge>;

template <typename Type, typename Judge>
MemoryReleaseFunction updateMemory(CachedMemoryHolder<Type, Judge>& cache,
    const Judge& judge) {
    if (&cache == nullptr)return {};
    return [&cache, judge](UniqueMemory memory, const size_t size) {
        MemorySpan<Type> data{std::move(memory), size};
        cache = std::make_shared<CachedValueHolder<MemorySpan<Type>, Judge>>(data, judge);
    };
}

template <typename Type, typename Judge>
class CacheRef final {
private:
    SharedCacheHolder<Type, Judge>* mRef;
    SharedCacheHolder<Type, Judge> mHolder;
    Judge mJudge;
public:
    CacheRef() : mRef(nullptr) {}

    CacheRef(SharedCacheHolder<Type, Judge>& ref, const Judge& judge)
        : mRef(&ref), mHolder(ref), mJudge(judge) {}

    operator bool() const {
        return static_cast<bool>(mHolder);
    }

    bool vaild() const {
        return mHolder->vaild(mJudge);
    }

    const Judge& getJudge() const {
        return mJudge;
    }

    void reset() {
        if (mRef)mRef->reset();
    }

    SharedCacheHolder<Type, Judge>& getRef() {
        return *mRef;
    }

    Type& getValue() {
        return mHolder->get();
    }
};
