#pragma once
#include <Math/Geometry.hpp>

class MaterialRef;

struct Interaction final {
    Transform toWorld;
    Point pos;
    Normal dir;
    Normal normal;
    vec2 uv;
    unsigned int id;//for ptex
    Normal dpdu, dpdv;
    Vector dndu, dndv;
    MaterialRef* material;

    Vector dpdx, dpdy;
    vec2 duvdx, duvdy;
};
