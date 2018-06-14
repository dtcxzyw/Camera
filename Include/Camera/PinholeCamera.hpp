#pragma once
#include <Math/Geometry.hpp>
#include <Camera/CamereSample.hpp>
#include <Sampler/Sampling.hpp>

class PinholeCameraRayGenerator final : RayGeneratorTag {
private:
    float mLensRadius, mFocalDistance;
    vec2 mScale, mOffset;
public:
    PinholeCameraRayGenerator(const float lensRadius, const float focalDistance,
        const vec2 scale, const vec2 offset)
        : mLensRadius(lensRadius * 1e-3f), mFocalDistance(focalDistance),
        mScale(scale), mOffset(offset) {}

    DEVICE Ray sample(const CameraSample& sample, float& weight) const {
        const vec2 pRaster = {sample.pFilm.x * 2.0f - 1.0f, sample.pFilm.y * -2.0f + 1.0f};
        const Vector pCamera{pRaster.x * mScale.x, pRaster.y * mScale.y, -1.0f};
        Ray ray{{}, pCamera};
        ray.xOri = ray.yOri = ray.origin;
        ray.xDir = {ray.dir.x + mOffset.x, ray.dir.y, ray.dir.z};
        ray.yDir = {ray.dir.x, ray.dir.y + mOffset.y, ray.dir.z};
        if (mLensRadius > 0.0f) {
            const auto pLens = mLensRadius * concentricSampleDisk(sample.pLens);
            ray.origin = Point{pLens.x, pLens.y, 0.0f};
            ray.dir = Point{ mFocalDistance * ray.dir } - ray.origin;
            ray.xDir = Point{ mFocalDistance * ray.xDir } -ray.origin;
            ray.yDir = Point{ mFocalDistance * ray.yDir } -ray.origin;
        }

        ray.dir = normalize(ray.dir);
        ray.xDir = normalize(ray.xDir);
        ray.yDir = normalize(ray.yDir);

        weight = 1.0f;
        return ray;
    }
};

//http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera
class PinholeCamera final {
public:
    enum class FitResolutionGate {
        Fill,
        Overscan
    } mode;

    float focalDistance; //in m
    float lensRadius;    //in mm
    float fov;
    vec2 filmGate; //in mm

    struct RasterPosConverter final {
        vec2 mul;
        float near, far;
    };

    PinholeCamera();
    RasterPosConverter toRasterPos(vec2 imageSize, float near, float far) const;

    //horizontal
    float toFov(float focalLength) const;
    float toFocalLength() const;

    PinholeCameraRayGenerator getRayGenerator(vec2 imageSize) const;
private:
    vec2 calcScale(vec2 imageSize, float near) const;
};
