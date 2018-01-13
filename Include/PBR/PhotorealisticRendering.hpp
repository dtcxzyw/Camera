#pragma once
#include <Base/Math.hpp>
constexpr auto inch2mm = 25.4f;

//http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera
struct Camera final {
    enum class FitResolutionGate {
        Fill, Overscan
    } mode;
    float focalLength;//in mm
    vec2 filmAperture;//in inches
    float near,far;
    struct RasterPosConverter final {
        vec2 mul;
        float near,far;
    };
    RasterPosConverter toRasterPos(vec2 imageSize) const {
        RasterPosConverter res;
        res.near = near,res.far=far;
        auto fratio = filmAperture.x / filmAperture.y;
        auto iratio = imageSize.x / imageSize.y;

        auto right = ((filmAperture.x*inch2mm / 2.0f) / focalLength) / near;
        auto top = ((filmAperture.y*inch2mm / 2.0f) / focalLength) / near;

        switch (mode) {
        case FitResolutionGate::Fill:
            if (fratio > iratio)right *= iratio / fratio;
            else top *= fratio / iratio;
            break;
        case FitResolutionGate::Overscan:
            if (fratio > iratio)top *= fratio / iratio;
            else right *= iratio / fratio;
            break;
        }

        res.mul = {near/right,near/top};
        return res;
    }

    //horizontal
    float toFOV() const {
        return 2.0f*atan((filmAperture.x*inch2mm / 2.0f) / focalLength);
    }
};

//https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch23.html
CUDAInline float DoFRadius(float z, float znear, float zfar, float f, float aperture, float df) {
    auto CoCScale = (aperture * f * df * (zfar - znear)) / ((df - f) * znear * zfar);
    auto CoCBias = (aperture * f * (znear - df)) / ((df * f) * znear);
    auto CoC = fabs(z * CoCScale + CoCBias);
    return CoC;
}
