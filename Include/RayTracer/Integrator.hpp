#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

class Transform;
class SceneDesc;
class RayGeneratorWrapper;
class FilmTile;
class CommandBuffer;

class Integrator : Uncopyable {
public:
    virtual void render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraTransform,
        const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, uvec2 offset, uvec2 dstSize) const = 0;
    virtual ~Integrator() = default;
};
