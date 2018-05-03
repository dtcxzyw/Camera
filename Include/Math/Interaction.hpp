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

CUDAINLINE void transform(const Transform& transform, Interaction& interaction) {
    interaction.pos = transform(interaction.pos);
    interaction.dir = transform(interaction.dir);
    interaction.normal = transform(interaction.normal);
    interaction.dpdu = transform(interaction.dpdu);
    interaction.dpdu = transform(interaction.dpdu);
    interaction.dndu = transform(interaction.dndu);
    interaction.dndv = transform(interaction.dndv);
}
