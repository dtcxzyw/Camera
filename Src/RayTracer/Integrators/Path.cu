#include <RayTracer/Integrators/Path.hpp>
#include <Camera/RayGeneratorWrapper.hpp>
#include <RayTracer/Film.hpp>
#include <Math/Geometry.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>
#include <Material/MaterialWrapper.hpp>
#include <Light/LightWrapper.hpp>
#include <Material/Material.hpp>
#include <RayTracer/Integrators/Utilities.hpp>

PathIntegrator::PathIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
    const unsigned int maxDepth, const unsigned int spp, const unsigned int launchSpp)
    : mSequenceGenerator(sequenceGenerator), mMaxDepth(maxDepth), mSpp(spp),
    mLaunchSpp(launchSpp) {}

static DEVICE Spectrum Li(RenderingContext& context, Ray ray, const unsigned int maxDepth) {
    Spectrum L{}, beta{1.0f};
    auto specularBounce = false;
    for (auto bounceCount = 0;; ++bounceCount) {
        Interaction interaction;
        const auto result = context.scene.intersect(ray, interaction);
        if (bounceCount == 0 | specularBounce) {
            Spectrum sum{};
            if (result) sum = interaction.le(-ray.dir);
            else {
                for (auto&& light : context.scene)
                    sum += light->le(ray);
            }
            L += beta * sum;
        }
        if (!result | bounceCount >= maxDepth)break;

        interaction.prepare(ray);
        Bsdf bsdf(interaction);
        interaction.material->computeScatteringFunctions(bsdf);
        //TODO:skip medium boundaries
        { }
        L += beta * uniformSampleOneLight(context, interaction, bsdf);
        const auto sampleF = bsdf.sampleF(-ray.dir, context.sample());
        if (sampleF.f.lum() <= 0.0f | sampleF.pdf <= 0.0f)break;
        beta *= sampleF.f * (fabs(dot(sampleF.wi, Vector{ interaction.shadingGeometry.normal })) / sampleF.pdf);
        specularBounce = static_cast<bool>(sampleF.type & BxDFType::Specular);
        ray = interaction.spawnRay(sampleF.wi);
        if (bounceCount > 3) {
            const auto q = fmax(0.05f, 1.0f - beta.maxComp());
            if (context.sample().x < q)break;
            beta /= 1.0f - q;
        }
    }
    return L;
}

static GLOBAL void renderKernel(const RayGeneratorWrapper rayGenerator,
    const Transform toWorld, const vec2 offset,
    const SequenceGenerator2DWrapper sequenceGenerator, const unsigned int seqOffset,
    FilmTileRef dst, const unsigned int maxDepth, const SceneRef scene, const unsigned int spp,
    const vec2 invDstSize) {
    const auto pid = blockIdx.z * blockIdx.z + threadIdx.x;
    if (pid >= spp)return;
    const auto id = ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * blockDim.x
        + threadIdx.x;
    RenderingContext context{sequenceGenerator, scene, seqOffset + (id << 5)};
    const auto tilePos = vec2{blockIdx.x, blockIdx.y} + context.sample();
    const CameraSample sample{(offset + tilePos) * invDstSize, context.sample()};
    float weight;
    const auto ray = rayGenerator.sample(sample, weight);
    if (weight > 0.0f) {
        dst.add(tilePos, Li(context, toWorld(ray), maxDepth) * weight);
    }
}

void PathIntegrator::render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraToWorld,
    const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2 offset, const uvec2 dstSize) const {
    const unsigned int blockSize = DeviceMonitor::get().getProp().maxThreadsPerBlock;
    const auto size = filmTile.size();
    constexpr auto stackSize = 2048U;
    auto todo = mSpp;
    while (todo) {
        const auto current = std::min(todo, mLaunchSpp);
        todo -= current;
        buffer.launchKernelDim(makeKernelDesc(renderKernel, stackSize),
            dim3{size.x, size.y, calcBlockSize(current, blockSize)},
            dim3{std::min(blockSize, current)}, rayGenerator, cameraToWorld, static_cast<vec2>(offset),
            mSequenceGenerator, todo * (offset.x << 16 | offset.y), filmTile.toRef(), mMaxDepth,
            scene.toRef(), current, 1.0f / vec2(dstSize));
    }
}
