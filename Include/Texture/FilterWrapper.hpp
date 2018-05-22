#pragma once
#include <Texture/Filter.hpp>

class FilterWrapper final {
private:
    enum class FilterClassType {
        BoxFilter,
        TriangleFilter,
        GaussianFilter,
        LanczosSincFilter
    };

    union {
        char unused{};
        BoxFilter dataBoxFilter;
        TriangleFilter dataTriangleFilter;
        GaussianFilter dataGaussianFilter;
        LanczosSincFilter dataLanczosSincFilter;
    };

    FilterClassType mType;
public:
    FilterWrapper(): mType(static_cast<FilterClassType>(15)) {};

    explicit FilterWrapper(const BoxFilter& data)
        : dataBoxFilter(data), mType(FilterClassType::BoxFilter) {}

    explicit FilterWrapper(const TriangleFilter& data)
        : dataTriangleFilter(data), mType(FilterClassType::TriangleFilter) {}

    explicit FilterWrapper(const GaussianFilter& data)
        : dataGaussianFilter(data), mType(FilterClassType::GaussianFilter) {}

    explicit FilterWrapper(const LanczosSincFilter& data)
        : dataLanczosSincFilter(data), mType(FilterClassType::LanczosSincFilter) {}

    FilterWrapper(const FilterWrapper& rhs) {
        memcpy(this, &rhs, sizeof(FilterWrapper));
    }

    FilterWrapper& operator=(const FilterWrapper& rhs) {
        memcpy(this, &rhs, sizeof(FilterWrapper));
    return *this;
    }

    DEVICE float evaluate(const vec2 pos) const {
        switch (mType) {
            case FilterClassType::BoxFilter: return dataBoxFilter.evaluate(pos);
            case FilterClassType::TriangleFilter: return dataTriangleFilter.evaluate(pos);
            case FilterClassType::GaussianFilter: return dataGaussianFilter.evaluate(pos);
            case FilterClassType::LanczosSincFilter: return dataLanczosSincFilter.evaluate(pos);
        }
    }

};
