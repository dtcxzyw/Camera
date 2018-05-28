#pragma once
#include <Math/Math.hpp>

struct CameraSample final {
    vec2 pFilm;
    vec2 pLens;
    DEVICE CameraSample(const vec2 pFilm, const vec2 pLens)
        : pFilm(pFilm), pLens(pLens) {}
};

struct RayGeneratorTag {};
