#pragma once
#include <Light/Shapes/Sphere.hpp>

class ShapeWrapper final {
private:
    enum class ShapeClassType {
        Sphere = 0
    };

    union {
        unsigned char unused{};
        Sphere dataSphere;
    };

    ShapeClassType mType;
public:
    ShapeWrapper(): mType(static_cast<ShapeClassType>(0xff)) {};

    explicit ShapeWrapper(const Sphere& data)
        : dataSphere(data), mType(ShapeClassType::Sphere) {}

    BOTH ShapeWrapper(const ShapeWrapper& rhs) {
        memcpy(this, &rhs, sizeof(ShapeWrapper));
    }

    BOTH ShapeWrapper& operator=(const ShapeWrapper& rhs) {
        memcpy(this, &rhs, sizeof(ShapeWrapper));
        return *this;
    }

    DEVICE Interaction sample(const Interaction& isect, const vec2 sample, float& pdf) const {
        switch (mType) {
            case ShapeClassType::Sphere: return dataSphere.sample(isect, sample, pdf);
        }
    }

    DEVICE float pdf(const Interaction& isect, const Vector& wi) const {
        switch (mType) {
            case ShapeClassType::Sphere: return dataSphere.pdf(isect, wi);
        }
    }

    DEVICE bool intersect(const Ray& ray) const {
        switch (mType) {
            case ShapeClassType::Sphere: return dataSphere.intersect(ray);
        }
    }

    DEVICE bool intersect(const Ray& ray, float& tHit, SurfaceInteraction& isect) const {
        switch (mType) {
            case ShapeClassType::Sphere: return dataSphere.intersect(ray, tHit, isect);
        }
    }

};
