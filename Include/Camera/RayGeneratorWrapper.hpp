#pragma once
#include <Camera/PinholeCamera.hpp>

class RayGeneratorWrapper final {
private:
    enum class RayGeneratorClassType {
        PinholeCameraRayGenerator
    };

    union {
        char unused{};
        PinholeCameraRayGenerator dataPinholeCameraRayGenerator;
    };

    RayGeneratorClassType mType;
public:
    RayGeneratorWrapper(): mType(static_cast<RayGeneratorClassType>(15)) {};

    explicit RayGeneratorWrapper(const PinholeCameraRayGenerator& data)
        : dataPinholeCameraRayGenerator(data), mType(RayGeneratorClassType::PinholeCameraRayGenerator) {}

    RayGeneratorWrapper(const RayGeneratorWrapper& rhs) {
        memcpy(this, &rhs, sizeof(RayGeneratorWrapper));
    }

    RayGeneratorWrapper& operator=(const RayGeneratorWrapper& rhs) {
        memcpy(this, &rhs, sizeof(RayGeneratorWrapper));
    return *this;
    }

    DEVICE Ray sample(const CameraSample& sample, float& weight) const {
        switch (mType) {
            case RayGeneratorClassType::PinholeCameraRayGenerator: return dataPinholeCameraRayGenerator.sample(sample, weight);
        }
    }

};
