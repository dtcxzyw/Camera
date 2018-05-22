#pragma once
#include <Material/Glass.hpp>
#include <Material/Metal.hpp>
#include <Material/Plastic.hpp>

class MaterialWrapper final {
private:
    enum class MaterialClassType {
        Glass,
        Metal,
        Plastic
    };

    union {
        char unused{};
        Glass dataGlass;
        Metal dataMetal;
        Plastic dataPlastic;
    };

    MaterialClassType mType;
public:
    MaterialWrapper(): mType(static_cast<MaterialClassType>(15)) {};

    explicit MaterialWrapper(const Glass& data)
        : dataGlass(data), mType(MaterialClassType::Glass) {}

    explicit MaterialWrapper(const Metal& data)
        : dataMetal(data), mType(MaterialClassType::Metal) {}

    explicit MaterialWrapper(const Plastic& data)
        : dataPlastic(data), mType(MaterialClassType::Plastic) {}

    MaterialWrapper(const MaterialWrapper& rhs) {
        memcpy(this, &rhs, sizeof(MaterialWrapper));
    }

    MaterialWrapper& operator=(const MaterialWrapper& rhs) {
        memcpy(this, &rhs, sizeof(MaterialWrapper));
    return *this;
    }

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, const TransportMode mode = TransportMode::Radiance) const {
        switch (mType) {
            case MaterialClassType::Glass: return dataGlass.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Metal: return dataMetal.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Plastic: return dataPlastic.computeScatteringFunctions(bsdf, mode);
        }
    }

};
