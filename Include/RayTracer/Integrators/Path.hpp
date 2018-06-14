#pragma once
#include <RayTracer/Integrator.hpp>
#include <Sampler/SequenceGenerator2DWrapper.hpp>

class PathIntegrator final : public Integrator {
private:
    SequenceGenerator2DWrapper mSequenceGenerator;
    uint32_t mMaxDepth, mSpp, mLaunchSpp;
public:
    PathIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
        uint32_t maxDepth, uint32_t spp, uint32_t launchSpp);
    void render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraTransform,
        const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, uvec2 offset, uvec2 dstSize) const override;
};
