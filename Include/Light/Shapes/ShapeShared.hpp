#pragma once
#include <Math/Geometry.hpp>
#include <Math/Interaction.hpp>

template <typename T>
class ShapeHelper {
private:
    DEVICE const T& self() const {
        return *static_cast<const T*>(this);
    }
public:
    Transform toWorld;
    bool reverseOri;

    explicit ShapeHelper(const Transform& trans, const bool reverseOri)
        : toWorld(trans), reverseOri(reverseOri) {}

    DEVICE Interaction sample(const Interaction& interaction, const vec2 sample, float& pdf) const {
        const auto isect = T::sample(sample, pdf);
        const auto wi = isect.pos - interaction.pos;
        if (glm::length2(wi) == 0.0f)pdf = 0.0f;
        else {
            wi = normalize(wi);
            pdf *= distance2(interaction.pos, isect.pos) /
                fabs(dot(Vector{isect.normal}, -wi));
            if (isinf(pdf)) pdf = 0.0f;
        }
        return isect;
    }

    DEVICE float pdf(const Interaction& interaction, const Vector& wi) const {
        const auto ray = interaction.spawnRay(wi);
        Interaction isectLight;
        auto tHit = ray.tMax;
        if (!self().intersect(ray, tHit, isectLight)) return 0.0f;

        const auto pdf = distance2(interaction.pos, isectLight.pos)*self().invArea() /
            fabs(dot(Vector{isectLight.localGeometry.normal}, -wi));
        return isinf(pdf) ? 0.0f : pdf;
    }
};

//TODO:Triangle Mesh Shape
