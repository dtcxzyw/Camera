#pragma once
#include <Math/Geometry.hpp>
#include <Math/EFloat.hpp>

class MaterialWrapper;

struct Interaction final {
    Transform toWorld;

    Point pos;
    Normal wo;
    Vector pError;

    Normal normal;
    vec2 uv;
    unsigned int id; //for ptex
    Normal dpdu, dpdv;
    Vector dndu, dndv;
    MaterialWrapper* material;

    Vector dpdx, dpdy;
    vec2 duvdx, duvdy;

    DEVICE void prepare(const Ray& ray) {
        normal = toWorld(normal);
        pos = toWorld(pos);

        const Vector n{normal};
        const auto d = dot(n, Vector{pos});
        const auto tx = (d - dot(n, Vector{ray.xOri})) / dot(n, ray.xDir);
        const auto ty = (d - dot(n, Vector{ray.xOri})) / dot(n, ray.yDir);
        const auto px = ray.origin + tx * ray.dir;
        const auto py = ray.origin + ty * ray.dir;
        dpdx = px - pos;
        dpdy = py - pos;

        constexpr unsigned int nxt[3] = {1, 2, 0};
        const auto dim = maxDim(n);
        const auto d0 = nxt[dim];
        const auto d1 = nxt[d0];

        const float a[2][2] = {{dpdu[d0], dpdv[d0]}, {dpdu[d1], dpdv[d1]}};
        const auto det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

        const float bx[2] = {px[d0] - pos[d0], px[d1] - pos[d1]};
        duvdx.s = (a[1][1] * bx[0] - a[0][1] * bx[1]) / det;
        duvdx.t = (a[0][0] * bx[1] - a[1][0] * bx[0]) / det;
        const float by[2] = {py[d0] - pos[d0], py[d1] - pos[d1]};
        duvdy.s = (a[1][1] * by[0] - a[0][1] * by[1]) / det;
        duvdy.t = (a[0][0] * by[1] - a[1][0] * by[0]) / det;
    }

    DEVICE Point calcOffsetOrigin(const Vector& w) const {
        const Vector n{normal};
        const auto d = dot(abs(n), pError);
        auto offset = d * n;
        if (dot(w, n) < 0) offset = -offset;
        auto po = pos + offset;
        #pragma unroll
        for (auto i = 0; i < 3; ++i) {
            if (offset[i] > 0)
                po[i] = nextFloatUp(po[i]);
            else if (offset[i] < 0)
                po[i] = nextFloatDown(po[i]);
        }
        return po;
    }

    DEVICE Ray spawnTo(const Point& dst) const {
        const auto d = dst - pos;
        return Ray{calcOffsetOrigin(d), d, 1.0f};
    }

    DEVICE Ray spawnRay(const Vector& w) const {
        return Ray{calcOffsetOrigin(w), w};
    }
};
