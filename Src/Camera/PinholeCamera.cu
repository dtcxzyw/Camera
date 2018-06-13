#include <Camera/PinholeCamera.hpp>

PinholeCamera::PinholeCamera() : mode(FitResolutionGate::Overscan), focalDistance(1e6f),
    lensRadius(0.0f), filmGate(24.892f, 18.669f), fov(90.0f) {}

PinholeCamera::RasterPosConverter PinholeCamera::toRasterPos(const vec2 imageSize, const float near,
    const float far) const {
    RasterPosConverter res;
    res.near = near, res.far = far;
    res.mul = calcScale(imageSize, near);
    return res;
}

float PinholeCamera::toFov(const float focalLength) const {
    return 2.0f * atan(filmGate.x / (2.0f * focalLength));
}

float PinholeCamera::toFocalLength() const {
    return filmGate.x / tan(glm::radians(fov) / 2.0f) * 0.5f;
}

PinholeCameraRayGenerator PinholeCamera::getRayGenerator(const vec2 imageSize) const {
    const auto scale = 1.0f / calcScale(imageSize, 1.0f);
    return {lensRadius, focalDistance, scale, vec2{2.0f, -2.0f} * scale / imageSize};
}

vec2 PinholeCamera::calcScale(const vec2 imageSize, const float near) const {
    const auto fratio = filmGate.x / filmGate.y;
    const auto iratio = imageSize.x / imageSize.y;

    const auto fac = 0.5f / (toFocalLength() * near);

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
