#pragma once
#include <Math/Geometry.hpp>
#include <Spectrum/SpectrumConfig.hpp>

class MaterialWrapper;
class LightWrapper;

struct Geometry {
    Normal normal;
    Vector dpdu, dpdv;
    Normal dndu, dndv;

    DEVICE void apply(const Transform& toWorld) {
        normal = normalize(toWorld(normal));
        dpdu = toWorld(dpdu), dpdv = toWorld(dpdv);
        dndu = toWorld(dndu), dndv = toWorld(dndv);
    }
};

struct Interaction {
    Point pos;
    Vector wo;
    Vector pError;

    Geometry localGeometry;

    DEVICE Point calcOffsetOrigin(const Vector& w) const {
        const Vector n{ localGeometry.normal };
        const auto d = dot(abs(n), pError);
        auto offset = d * n;
        if (dot(w, n) < 0.0f) offset = -offset;
        auto po = pos + offset;
#pragma unroll
        for (auto i = 0; i < 3; ++i) {
            if (offset[i] > 0.0f)
                po[i] = nextFloatUp(po[i]);
            else if (offset[i] < 0.0f)
                po[i] = nextFloatDown(po[i]);
        }
        return po;
    }

    DEVICE Ray spawnTo(const Point& dst) const {
        constexpr auto shadowDis = 1.0f - 1e-5f;
        const auto d = dst - pos;
        const auto ori = calcOffsetOrigin(d);
        return Ray{ ori, d, shadowDis };
    }

    DEVICE Ray spawnRay(const Vector& w) const {
        return Ray{ calcOffsetOrigin(w), w };
    }

    DEVICE void applyTransform(const Transform& toWorld) {
        Vector err;
        pos = toWorld(pos, pError, err);
        pError = err;
        wo = normalize(toWorld(wo));
        localGeometry.apply(toWorld);
    }
};

struct SurfaceInteraction final :public Interaction {
    Geometry shadingGeometry;

    vec2 uv;
    uint32_t id; //for ptex
    MaterialWrapper* material;
    LightWrapper* areaLight;

    Vector dpdx, dpdy;
    vec2 duvdx, duvdy;

    DEVICE SurfaceInteraction() :id(0), material(nullptr), areaLight(nullptr) {}

    DEVICE void applyTransform(const Transform& toWorld) {
        Interaction::applyTransform(toWorld);

        shadingGeometry.apply(toWorld);
        shadingGeometry.normal = faceForward(shadingGeometry.normal, localGeometry.normal);
    }

    DEVICE void computeDifferentials(const RayDifferential& ray) {

        if (ray.hasDifferentials) {
            const Vector n{ localGeometry.normal };
            const auto d = dot(n, Vector{ pos });
            const auto tx = (d - dot(n, Vector{ ray.xOri })) / dot(n, ray.xDir);
            const auto ty = (d - dot(n, Vector{ ray.xOri })) / dot(n, ray.yDir);
            const auto px = ray.origin + tx * ray.dir;
            const auto py = ray.origin + ty * ray.dir;
            dpdx = px - pos;
            dpdy = py - pos;

            constexpr int nxt[3] = { 1, 2, 0 };
            const auto dim = maxDim(n);
            const auto d0 = nxt[dim];
            const auto d1 = nxt[d0];

            const float a[2][2] = {
                {localGeometry.dpdu[d0], localGeometry.dpdv[d0]},
                {localGeometry.dpdu[d1], localGeometry.dpdv[d1]}
            };
            const auto det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            const auto invDet = 1.0f / det;

            const float bx[2] = { px[d0] - pos[d0], px[d1] - pos[d1] };
            duvdx.s = (a[1][1] * bx[0] - a[0][1] * bx[1]) * invDet;
            duvdx.t = (a[0][0] * bx[1] - a[1][0] * bx[0]) * invDet;
            const float by[2] = { py[d0] - pos[d0], py[d1] - pos[d1] };
            duvdy.s = (a[1][1] * by[0] - a[0][1] * by[1]) * invDet;
            duvdy.t = (a[0][0] * by[1] - a[1][0] * by[0]) * invDet;
        }
    }

    DEVICE Spectrum le(const Vector& w) const;
};
