#pragma once
#include <RayTracer/Integrator.hpp>
#include <Sampler/SequenceGenerator2DWrapper.hpp>

class WhittedIntegrator final : public Integrator {
private:
    SequenceGenerator2DWrapper mSequenceGenerator;
    uint32_t mMaxDepth, mSpp;
public:
    WhittedIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
        uint32_t maxDepth, uint32_t spp);
    void render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraTransform,
        const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, uvec2 offset, uvec2 dstSize) const override;
};
