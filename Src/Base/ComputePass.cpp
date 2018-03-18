#include <Base/ComputePass.hpp>

CachedValue::CachedValue(const Id id) :mId(id) {}
Id CachedValue::getId() const {
    return mId;
}

bool CachedValue::operator==(const CachedValue& val) const {
    return mId == val.mId;
}
