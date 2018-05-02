#pragma once
#include <Math/Geometry.hpp>

class MaterialRef;

struct Interaction final {
    Point pos;
    Normal dir;
    Normal normal;
    vec2 uv;
    unsigned int id;//for ptex
    Normal dpdu, dpdv;
    Vector dndu, dndv;
    MaterialRef* material;
};

CUDAINLINE void transform(const Transform& trans, Interaction& info) {
    info.pos = trans(info.pos);
    info.dir = trans(info.dir);
    info.normal = trans(info.normal);
    info.dpdu = trans(info.dpdu);
    info.dpdv = trans(info.dpdv);
    info.dndu = trans(info.dndu);
    info.dndv = trans(info.dndv);
}
