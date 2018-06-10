#pragma once
#include <Sampler/SequenceGenerator2DWrapper.hpp>
#include <RayTracer/Scene.hpp>
#include <Material/Material.hpp>

struct RenderingContext {
    const SequenceGenerator2DWrapper sequenceGenerator;
    const SceneRef scene;
    unsigned int offset;

    DEVICE RenderingContext(const SequenceGenerator2DWrapper& sequenceGenerator,
        const SceneRef& scene, const unsigned int offset)
        : sequenceGenerator(sequenceGenerator), scene(scene), offset(offset) {}

    DEVICE vec2 sample() {
        return sequenceGenerator.sample(++offset);
    }
};

DEVICE Spectrum uniformSampleOneLight(RenderingContext& context, const Interaction& interaction,
    const Bsdf& bsdf);
