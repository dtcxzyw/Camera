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
    const uint32_t maxDepth, const uint32_t spp, const uint32_t launchSpp,const float threshold)
    : mSequenceGenerator(sequenceGenerator), mMaxDepth(maxDepth), mSpp(spp),
    mLaunchSpp(launchSpp), mThreshold(threshold) {}

static DEVICE Spectrum Li(RenderingContext& context, RayDifferential ray,
    const uint32_t maxDepth,const float threshold) {
    Spectrum L{}, beta{1.0f};
    auto etaScale = 1.0f;
    auto specularBounce = false;
    for (auto bounceCount = 0;; ++bounceCount) {
        SurfaceInteraction interaction;
        const auto result = context.scene.intersect(ray, interaction);
        if (bounceCount == 0 | specularBounce) {
            Spectrum sum{};
            if (result) sum = interaction.le(-ray.dir);
            else {
                for (auto&& light : context.scene)
                    sum += light.le(ray);
            }
            L += beta * sum;
        }
        if (!result | bounceCount >= maxDepth | interaction.areaLight != nullptr)break;

        interaction.computeDifferentials(ray);
        Bsdf bsdf(interaction);
        if (interaction.material)interaction.material->computeScatteringFunctions(bsdf);

        if (bsdf.match(BxDFType::All^BxDFType::Specular)) {
            L += beta * uniformSampleOneLight(context, interaction, bsdf,
                context.scene.lookUp(interaction.pos));
        }
        const auto sampleF = bsdf.sampleF(-ray.dir, context.sample());
        if (sampleF.f.y() <= 0.0f | sampleF.pdf <= 0.0f)break;
        beta *= sampleF.f * (fabs(dot(sampleF.wi, Vector{interaction.shadingGeometry.normal})) / sampleF.pdf);
        specularBounce = static_cast<bool>(sampleF.type & BxDFType::Specular);

        if (static_cast<bool>(sampleF.type & BxDFType::Reflection) & 
            static_cast<bool>(sampleF.type & BxDFType::Transmission)) {
            const auto eta = bsdf.getEta();
            etaScale *= dot(-ray.dir, Vector{ interaction.localGeometry.normal }) > 0.0f ?
                eta * eta : 1.0f / (eta * eta);
        }

        ray = RayDifferential{ interaction.spawnRay(sampleF.wi) };

        //TODO:BSSRDF

        const auto scaledBeta = beta * etaScale;
        if (bounceCount > 3 & scaledBeta.maxComp() < threshold) {
            const auto q = fmax(0.05f, 1.0f - scaledBeta.maxComp());
            if (context.sample().x < q)break;
            beta /= 1.0f - q;
        }
    }
    return L;
}

static GLOBAL void renderKernel(const RayGeneratorWrapper rayGenerator,
    const Transform toWorld, const vec2 offset,
    const SequenceGenerator2DWrapper sequenceGenerator, const uint32_t seqOffset,
    FilmTileRef dst, const uint32_t maxDepth, const SceneRef scene, const uint32_t spp,
    const vec2 invDstSize,const float threshold) {
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
        dst.add(tilePos, Li(context, toWorld(ray), maxDepth, threshold) * weight);
    }
}

void PathIntegrator::render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraToWorld,
    const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2 offset, const uvec2 dstSize) const {
    const uint32_t blockSize = DeviceMonitor::get().getProp().maxThreadsPerBlock;
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
            scene.toRef(), current, 1.0f / vec2(dstSize), mThreshold);
    }
}
