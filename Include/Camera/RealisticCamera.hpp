#pragma once
#include <Core/Memory.hpp>
#include <Camera/CamereSample.hpp>
#include <string>
#include <Math/Geometry.hpp>

struct LensElementInterface {
    float curvatureRadius;
    float thickness;
    float eta;
    float apertureRadius;
};

class RealisticCameraRayGenerator final /*:public RayGeneratorTag*/ {
private:
public:
    DEVICE float generateRay(const CameraSample& sample, RayDifferential& ray) {
        return 0.0f;
    }
};

class RealisticCamera final {
private:
    MemorySpan<LensElementInterface> mLens;
    float mApertureDiameter, mFocalDistance;
public:
    RealisticCamera(const std::string& lensDesc, float apertureDiameter, float focalDistance);
};
