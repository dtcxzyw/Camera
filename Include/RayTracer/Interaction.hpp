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
