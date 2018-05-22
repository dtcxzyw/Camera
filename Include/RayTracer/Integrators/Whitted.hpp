#pragma once
#include <RayTracer/Integrator.hpp>
#include <Sampler/SequenceGenerator2DWrapper.hpp>
#include <Core/DispatchSystem.hpp>

class WhittedIntegrator final :public Integrator {
private:
    SequenceGenerator2DWrapper mSequenceGenerator;
    unsigned int mMaxDepth, mSpp, mSeqOffset;
public:
    WhittedIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
        unsigned int maxDepth,unsigned int spp,unsigned int seqOffset);
    Future render(SceneDesc& scene, const Transform& cameraTransform,
        const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2& offset) const override;
};
