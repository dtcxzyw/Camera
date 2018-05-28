#include <Camera/PinholeCamera.hpp>

PinholeCamera::PinholeCamera() : mode(FitResolutionGate::Overscan), focalLength(15.0f),
    lensRadius(0.0f), filmGate(24.892f, 18.669f), near(0.1f), far(1e4f) {}

PinholeCamera::RasterPosConverter PinholeCamera::toRasterPos(const vec2 imageSize) const {
    RasterPosConverter res;
    res.near = near, res.far = far;
    res.mul = calcScale(imageSize);
    return res;
}

float PinholeCamera::toFov() const {
    return 2.0f * atan((filmGate.x * 0.5f) / focalLength);
}

PinholeCameraRayGenerator PinholeCamera::getRayGenerator(const vec2 imageSize) const {
    const auto scale = vec2{1.0f, -1.0f} / calcScale(imageSize);
    return { lensRadius,focalLength, scale, 2.0f * scale / imageSize };
}

vec2 PinholeCamera::calcScale(const vec2 imageSize) const {
    const auto fratio = filmGate.x / filmGate.y;
    const auto iratio = imageSize.x / imageSize.y;
    const auto fac = 0.5f / (focalLength * near);

    auto right = filmGate.x * fac;
    auto top = filmGate.y * fac;

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

    return {near / right, near / top};
}
