#pragma once
#include <Camera/PinholeCamera.hpp>

class RayGeneratorWrapper final {
private:
    enum class RayGeneratorClassType {
        PinholeCameraRayGenerator = 0
    };

    union {
        unsigned char unused{};
        PinholeCameraRayGenerator dataPinholeCameraRayGenerator;
    };

    RayGeneratorClassType mType;
public:
    RayGeneratorWrapper(): mType(static_cast<RayGeneratorClassType>(0xff)) {};

    explicit RayGeneratorWrapper(const PinholeCameraRayGenerator& data)
        : dataPinholeCameraRayGenerator(data), mType(RayGeneratorClassType::PinholeCameraRayGenerator) {}

    BOTH RayGeneratorWrapper(const RayGeneratorWrapper& rhs) {
        memcpy(this, &rhs, sizeof(RayGeneratorWrapper));
    }

    BOTH RayGeneratorWrapper& operator=(const RayGeneratorWrapper& rhs) {
        memcpy(this, &rhs, sizeof(RayGeneratorWrapper));
        return *this;
    }

    DEVICE RayDifferential sample(const CameraSample& sample, float& weight) const {
        switch (mType) {
            case RayGeneratorClassType::PinholeCameraRayGenerator: return dataPinholeCameraRayGenerator.sample(sample, weight);
        }
    }

};
