#pragma once
#include <Sampler/SequenceGenerator.hpp>

class SequenceGenerator1DWrapper final {
private:
    enum class SequenceGenerator1DClassType {
        RadicalInverse = 0,
        Sobol1D = 1
    };

    union {
        unsigned char unused{};
        RadicalInverse dataRadicalInverse;
        Sobol1D dataSobol1D;
    };

    SequenceGenerator1DClassType mType;
public:
    SequenceGenerator1DWrapper(): mType(static_cast<SequenceGenerator1DClassType>(0xff)) {};

    explicit SequenceGenerator1DWrapper(const RadicalInverse& data)
        : dataRadicalInverse(data), mType(SequenceGenerator1DClassType::RadicalInverse) {}

    explicit SequenceGenerator1DWrapper(const Sobol1D& data)
        : dataSobol1D(data), mType(SequenceGenerator1DClassType::Sobol1D) {}

    BOTH SequenceGenerator1DWrapper(const SequenceGenerator1DWrapper& rhs) {
        memcpy(this, &rhs, sizeof(SequenceGenerator1DWrapper));
    }

    BOTH SequenceGenerator1DWrapper& operator=(const SequenceGenerator1DWrapper& rhs) {
        memcpy(this, &rhs, sizeof(SequenceGenerator1DWrapper));
        return *this;
    }

    DEVICE float sample(const unsigned int index) const {
        switch (mType) {
            case SequenceGenerator1DClassType::RadicalInverse: return dataRadicalInverse.sample(index);
            case SequenceGenerator1DClassType::Sobol1D: return dataSobol1D.sample(index);
        }
    }

};
