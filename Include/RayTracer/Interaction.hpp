#pragma once
#include <Math/Geometry.hpp>

struct Interaction final {
    Point pos;
    Normal dir;
    Normal normal;
    vec2 uv;
    Normal dpdu, dpdv;
    Vector dndu, dndv;
};
