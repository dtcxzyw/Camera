#pragma once
#include <RayTracer/Integrator.hpp>
#include <Sampler/SequenceGenerator2DWrapper.hpp>

class WhittedIntegrator final : public Integrator {
private:
    SequenceGenerator2DWrapper mSequenceGenerator;
    unsigned int mMaxDepth, mSpp;
public:
    WhittedIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
        unsigned int maxDepth, unsigned int spp);
    void render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraTransform,
        const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, uvec2 offset, uvec2 dstSize) const override;
};
