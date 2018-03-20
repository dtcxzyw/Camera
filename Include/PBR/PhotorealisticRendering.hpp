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
    RasterPosConverter toRasterPos(const vec2 imageSize) const {
        RasterPosConverter res;
        res.near = near,res.far=far;
        const auto fratio = filmAperture.x / filmAperture.y;
        const auto iratio = imageSize.x / imageSize.y;

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
    float toFov() const {
        return 2.0f*atan((filmAperture.x*inch2mm / 2.0f) / focalLength);
    }
};

//https://developer.nvidia.com/gpugems/RefGems/gpugems_ch23.html
CUDAINLINE float DoFRadius(const float z, const float znear, const float zfar, const float f, 
    const float aperture, const float df) {
    const auto CoCScale = (aperture * f * df * (zfar - znear)) / ((df - f) * znear * zfar);
    const auto CoCBias = (aperture * f * (znear - df)) / ((df * f) * znear);
    const auto CoC = fabs(z * CoCScale + CoCBias);
    return CoC;
}
