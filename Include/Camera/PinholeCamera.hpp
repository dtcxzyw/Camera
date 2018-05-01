#pragma once
#include <Math/Geometry.hpp>
#include <Camera/CamereSample.hpp>
#include <Sampler/Samping.hpp>

//http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera
class PinholeCamera final {
public:
    enum class FitResolutionGate {
        Fill,
        Overscan
    } mode;

    float focalLength,lensRadius; //in mm
    vec2 filmGate; //in mm
    float near, far;

    struct RasterPosConverter final {
        vec2 mul;
        float near, far;
    };

    struct RayGenerator final {
        float lensRadius, focalLength;
        vec2 scale,offset;
        //offset=2.0f*scale/screenSize
        CUDA Ray generateRay(const CameraSample& sample) const {
            const auto pRaster = sample.pFilm*2.0f - 1.0f;
            const Point pCamera = { pRaster.x*scale.x,pRaster.y*scale.y,-1.0f };
            Ray ray{ {},Vector(pCamera) };
            if(lensRadius>0.0f) {
                const auto pLens = lensRadius * concentricSampleDisk(sample.pLens);
                const auto focus = ray(focalLength);
                ray.origin = Point{ pLens.x,pLens.y,0.0f };
                ray.dir = focus - ray.origin;
                ray.xDir = ray.dir + Vector{ offset.x*focalLength, 0.0f, 0.0f };
                ray.yDir = ray.dir + Vector{ 0.0f,offset.y*focalLength,0.0f };
            }
            else {
                ray.xDir = { ray.dir.x + offset.x,ray.dir.y,ray.dir.z };
                ray.yDir = { ray.dir.x,ray.dir.y + offset.y,ray.dir.z };
            }

            ray.xOri = ray.yOri = ray.origin;

            return ray;
        }
    };

    PinholeCamera();
    RasterPosConverter toRasterPos(vec2 imageSize) const;

    //horizontal
    float toFov() const;

    RayGenerator getRayGenerator(vec2 imageSize) const;
private:
    vec2 calcScale(vec2 imageSize) const;
};
