#pragma once
#include <Material/Glass.hpp>
#include <Material/Matte.hpp>
#include <Material/Metal.hpp>
#include <Material/Mirror.hpp>
#include <Material/Plastic.hpp>
#include <Material/Substrate.hpp>

class MaterialWrapper final {
private:
    enum class MaterialClassType {
        Glass = 0,
        Matte = 1,
        Metal = 2,
        Mirror = 3,
        Plastic = 4,
        Subtrate = 5
    };

    union {
        unsigned char unused{};
        Glass dataGlass;
        Matte dataMatte;
        Metal dataMetal;
        Mirror dataMirror;
        Plastic dataPlastic;
        Subtrate dataSubtrate;
    };

    MaterialClassType mType;
public:
    MaterialWrapper(): mType(static_cast<MaterialClassType>(0xff)) {};

    explicit MaterialWrapper(const Glass& data)
        : dataGlass(data), mType(MaterialClassType::Glass) {}

    explicit MaterialWrapper(const Matte& data)
        : dataMatte(data), mType(MaterialClassType::Matte) {}

    explicit MaterialWrapper(const Metal& data)
        : dataMetal(data), mType(MaterialClassType::Metal) {}

    explicit MaterialWrapper(const Mirror& data)
        : dataMirror(data), mType(MaterialClassType::Mirror) {}

    explicit MaterialWrapper(const Plastic& data)
        : dataPlastic(data), mType(MaterialClassType::Plastic) {}

    explicit MaterialWrapper(const Subtrate& data)
        : dataSubtrate(data), mType(MaterialClassType::Subtrate) {}

    BOTH MaterialWrapper(const MaterialWrapper& rhs) {
        memcpy(this, &rhs, sizeof(MaterialWrapper));
    }

    BOTH MaterialWrapper& operator=(const MaterialWrapper& rhs) {
        memcpy(this, &rhs, sizeof(MaterialWrapper));
        return *this;
    }

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, const TransportMode mode = TransportMode::Radiance) const {
        switch (mType) {
            case MaterialClassType::Glass: return dataGlass.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Matte: return dataMatte.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Metal: return dataMetal.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Mirror: return dataMirror.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Plastic: return dataPlastic.computeScatteringFunctions(bsdf, mode);
            case MaterialClassType::Subtrate: return dataSubtrate.computeScatteringFunctions(bsdf, mode);
        }
    }

};
