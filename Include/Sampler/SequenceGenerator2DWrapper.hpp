#pragma once
#include <Sampler/SequenceGenerator.hpp>

class SequenceGenerator2DWrapper final {
private:
    enum class SequenceGenerator2DClassType {
        Halton2D,
        Sobol2D
    };

    union {
        char unused{};
        Halton2D dataHalton2D;
        Sobol2D dataSobol2D;
    };

    SequenceGenerator2DClassType mType;
public:
    SequenceGenerator2DWrapper(): mType(static_cast<SequenceGenerator2DClassType>(15)) {};

    explicit SequenceGenerator2DWrapper(const Halton2D& data)
        : dataHalton2D(data), mType(SequenceGenerator2DClassType::Halton2D) {}

    explicit SequenceGenerator2DWrapper(const Sobol2D& data)
        : dataSobol2D(data), mType(SequenceGenerator2DClassType::Sobol2D) {}

    SequenceGenerator2DWrapper(const SequenceGenerator2DWrapper& rhs) {
        memcpy(this, &rhs, sizeof(SequenceGenerator2DWrapper));
    }

    SequenceGenerator2DWrapper& operator=(const SequenceGenerator2DWrapper& rhs) {
        memcpy(this, &rhs, sizeof(SequenceGenerator2DWrapper));
    return *this;
    }

    DEVICE vec2 sample(const unsigned int index) const {
        switch (mType) {
            case SequenceGenerator2DClassType::Halton2D: return dataHalton2D.sample(index);
            case SequenceGenerator2DClassType::Sobol2D: return dataSobol2D.sample(index);
        }
    }

};
