#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <Core/DispatchSystem.hpp>

class Transform;
class SceneDesc;
class RayGeneratorWrapper;
class FilmTile;

class Integrator :Uncopyable {
public:
    virtual Future render(SceneDesc& scene, const Transform& cameraTransform,
        const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2& offset) const = 0;
    virtual ~Integrator() = default;
};
