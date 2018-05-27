#pragma once
#include <Math/Interaction.hpp>

template <typename TexCoord>
struct TextureMappingInfo final {
    TexCoord pos, dpdx, dpdy;

    DEVICE TextureMappingInfo(const TexCoord pos, const TexCoord dpdx, const TexCoord dpdy)
        : pos(pos), dpdx(dpdx), dpdy(dpdy) {}
};

using TextureMapping2DInfo = TextureMappingInfo<vec2>;
using TextureMapping3DInfo = TextureMappingInfo<Vector>;

struct TextureMapping2D {};

class UVMapping final : TextureMapping2D {
private:
    vec2 mScale, mOffset;
public:
    UVMapping(const vec2 scale, const vec2 offset) : mScale(scale), mOffset(offset) {}
    DEVICE TextureMapping2DInfo map(const Interaction& interaction) const {
        return {interaction.uv * mScale + mOffset, interaction.duvdx * mScale, interaction.duvdy * mScale};
    }
};

namespace Impl {
    DEVICEINLINE float clampDiff(const float x) {
        const auto absx = fabs(x);
        return copysign(fmin(absx, 1.0f - absx), x);
    }

    DEVICEINLINE vec2 fixTexCoordDiff(const vec2 diff) {
        return {clampDiff(diff.x), clampDiff(diff.y)};
    }
}

class SphericalMapping final : TextureMapping2D {
private:
    Transform mTransform;
    DEVICE vec2 sphere(const Point& pos) const {
        const auto p = normalize(Vector(mTransform(pos)));
        return {acos(p.z) * one_over_pi<float>() + 0.5f, atan2(p.y, p.x) * one_over_two_pi<float>() + 0.5f};
    }

public:
    explicit SphericalMapping(const Transform& transform) : mTransform(transform) {}
    DEVICE TextureMapping2DInfo map(const Interaction& interaction) const {
        const auto uv = sphere(interaction.pos);
        constexpr auto delta = 0.1f, fac = 1.0f / delta;
        return {
            uv, Impl::fixTexCoordDiff((sphere(interaction.pos + delta * interaction.dpdx) - uv) * fac),
            Impl::fixTexCoordDiff((sphere(interaction.pos + delta * interaction.dpdy) - uv) * fac)
        };
    }
};

class CylindricalMapping final : TextureMapping2D {
private:
    Transform mTransform;
    DEVICE vec2 cylinder(const Point& pos) const {
        const auto p = normalize(Vector(mTransform(pos)));
        return {atan2(p.y, p.x) * one_over_two_pi<float>() + 0.5f, p.z};
    }

public:
    explicit CylindricalMapping(const Transform& transform) : mTransform(transform) {}
    DEVICE TextureMapping2DInfo map(const Interaction& interaction) const {
        const auto uv = cylinder(interaction.pos);
        constexpr auto delta = 0.1f, fac = 1.0f / delta;
        return {
            uv, Impl::fixTexCoordDiff((cylinder(interaction.pos + delta * interaction.dpdx) - uv) * fac),
            Impl::fixTexCoordDiff((cylinder(interaction.pos + delta * interaction.dpdy) - uv) * fac)
        };
    }
};

class PlanarMapping final : TextureMapping2D {
private:
    Transform mTransform;
    Vector mS, mT;
    vec2 mOffset;
public:
    PlanarMapping(const Transform& transform, const Vector& s, const Vector& t, const vec2 offset)
        : mTransform(transform), mS(s), mT(t), mOffset(offset) {}

    DEVICE TextureMapping2DInfo map(const Interaction& interaction) const {
        const Vector p(interaction.pos);
        return {
            mOffset + vec2{dot(p, mS), dot(p, mT)},
            vec2{dot(interaction.dpdx, mS), dot(interaction.dpdx, mT)},
            vec2{dot(interaction.dpdy, mS), dot(interaction.dpdy, mT)}
        };
    }
};

class TextureMapping3D{
private:
    Transform mTransform;
public:
    explicit TextureMapping3D(const Transform& transform) : mTransform(transform) {}
    DEVICE TextureMapping3DInfo map(const Interaction& interaction) const {
        return {
            Vector(mTransform(interaction.pos)), mTransform(interaction.dpdx),
            mTransform(interaction.dpdy)
        };
    }
};
