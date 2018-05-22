#include <RayTracer/Integrators/Whitted.hpp>
#include <RayTracer/Scene.hpp>
#include <Camera/RayGeneratorWrapper.hpp>
#include <RayTracer/Film.hpp>
#include <Math/Geometry.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>
#include <Material/MaterialWrapper.hpp>
#include <Light/LightWrapper.hpp>

WhittedIntegrator::WhittedIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
    const unsigned int maxDepth, const unsigned int spp, const unsigned int seqOffset)
    : mSequenceGenerator(sequenceGenerator), mMaxDepth(maxDepth), mSpp(spp),
    mSeqOffset(seqOffset) {}

struct RenderingContext {
    const SequenceGenerator2DWrapper& sequenceGenerator;
    const SceneRef& scene;
    unsigned int offset;

    RenderingContext(const SequenceGenerator2DWrapper& sequenceGenerator,
        const SceneRef& scene, const unsigned int offset)
        : sequenceGenerator(sequenceGenerator), scene(scene), offset(offset) {}

    vec2 sample() {
        return sequenceGenerator.sample(++offset);
    }
};

DEVICE Spectrum specularReflect() {
    NOT_IMPLEMENTED();
}

DEVICE Spectrum specularTransmit() {
    NOT_IMPLEMENTED();
}

DEVICE Spectrum Li(RenderingContext& context, const Ray& ray, const unsigned int depth) {
    Interaction interaction;
    if (context.scene.intersect(ray, interaction)) {
        interaction.prepare(ray);
        Bsdf bsdf(interaction);
        interaction.material->computeScatteringFunctions(bsdf);
        Spectrum L{};
        for (auto&& light : context.scene) {
            const auto sample = light->sampleLi(context.sample(), interaction.pos);
            if (sample.illumination.lum() > 0.0f & sample.pdf > 0.0f 
                & !context.scene.intersect(interaction.spawnTo(sample.src))) {
                const auto f = bsdf.f(Vector{ interaction.dir }, Vector{ sample.wi });
                if (f.lum() > 0.0f)L += f * sample.illumination 
                    * fabs(dot(sample.wi, interaction.normal)) / sample.pdf;
            }
        }
        if (depth) {
            L += specularReflect();
            L += specularTransmit();
        }
        return L;
    }
    Spectrum L{};
    for (auto&& light : context.scene)
        L += light->le(ray);
    return L;
}

GLOBAL void renderKernel(const RayGeneratorWrapper& rayGenerator,
    const Transform& toWorld, const vec2 offset,
    const SequenceGenerator2DWrapper& sequenceGenerator, const unsigned int seqOffset,
    FilmTileRef& dst, const unsigned int maxDepth, const SceneRef& scene) {
    const auto id = ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * blockDim.x
        + threadIdx.x;
    RenderingContext context{sequenceGenerator, scene, seqOffset + (id << 5)};
    const auto tilePos = vec2{blockIdx.x, blockIdx.y} + context.sample();
    const CameraSample sample{offset + tilePos, context.sample()};
    float weight;
    const auto ray = rayGenerator.sample(sample, weight);
    if (weight <= 0.0f)return;
    dst.add(tilePos, Li(context,toWorld(ray), maxDepth) * weight);
}

Future WhittedIntegrator::render(SceneDesc& scene, const Transform& cameraTransform,
    const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2& offset) const {
    auto buffer = std::make_unique<CommandBuffer>();
    const unsigned int blockSize = DeviceMonitor::get().getProp().maxThreadsPerBlock;
    {
        const auto size = filmTile.size();
        buffer->launchKernelDim(renderKernel, dim3{size.x, size.y, calcBlockSize(mSpp, blockSize)},
            dim3{ blockSize }, rayGenerator, inverse(cameraTransform), static_cast<vec2>(offset), 
            mSequenceGenerator, mSeqOffset, filmTile.toRef(), mMaxDepth, scene.toRef());
    }
    return Environment::get().submit(std::move(buffer));
}
