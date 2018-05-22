#pragma once
#include <Math/Geometry.hpp>
#include <Camera/CamereSample.hpp>
#include <Sampler/Samping.hpp>

class PinholeCameraRayGenerator final : RayGeneratorTag {
private:
    float mLensRadius, mFocalLength;
    vec2 mScale, mOffset;
public:
    PinholeCameraRayGenerator(const float lensRadius, const float focalLength,
        const vec2 scale, const vec2 offset)
        : mLensRadius(lensRadius), mFocalLength(focalLength), mScale(scale), mOffset(offset) {}

    //offset=2.0f*scale/screenSize
    DEVICE Ray sample(const CameraSample& sample, float& weight) const {
        const auto pRaster = sample.pFilm * 2.0f - 1.0f;
        const Point pCamera = {pRaster.x * mScale.x, pRaster.y * mScale.y, -1.0f};
        Ray ray{{}, Vector(pCamera)};
        if (mLensRadius > 0.0f) {
            const auto pLens = mLensRadius * concentricSampleDisk(sample.pLens);
            const auto focus = ray(mFocalLength);
            ray.origin = Point{pLens.x, pLens.y, 0.0f};
            ray.dir = focus - ray.origin;
            ray.xDir = ray.dir + Vector{mOffset.x * mFocalLength, 0.0f, 0.0f};
            ray.yDir = ray.dir + Vector{0.0f, mOffset.y * mFocalLength, 0.0f};
        }
        else {
            ray.xDir = {ray.dir.x + mOffset.x, ray.dir.y, ray.dir.z};
            ray.yDir = {ray.dir.x, ray.dir.y + mOffset.y, ray.dir.z};
        }

        ray.xOri = ray.yOri = ray.origin;

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

    float focalLength, lensRadius; //in mm
    vec2 filmGate;                 //in mm
    float near, far;

    struct RasterPosConverter final {
        vec2 mul;
        float near, far;
    };

    PinholeCamera();
    RasterPosConverter toRasterPos(vec2 imageSize) const;

    //horizontal
    float toFov() const;

    PinholeCameraRayGenerator getRayGenerator(vec2 imageSize) const;
private:
    vec2 calcScale(vec2 imageSize) const;
};
