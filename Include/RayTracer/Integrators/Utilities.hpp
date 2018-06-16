#pragma once
#include <Sampler/SequenceGenerator2DWrapper.hpp>
#include <RayTracer/Scene.hpp>
#include <Material/Material.hpp>

struct RenderingContext {
    const SequenceGenerator2DWrapper sequenceGenerator;
    const SceneRef scene;
    uint32_t offset;

    DEVICE RenderingContext(const SequenceGenerator2DWrapper& sequenceGenerator,
        const SceneRef& scene, const uint32_t offset)
        : sequenceGenerator(sequenceGenerator), scene(scene), offset(offset) {}

    DEVICE vec2 sample() {
        return sequenceGenerator.sample(++offset);
    }
};

DEVICE Spectrum uniformSampleOneLight(RenderingContext& context, const Interaction& interaction,
    const Bsdf& bsdf,const LightDistribution* distribution);
