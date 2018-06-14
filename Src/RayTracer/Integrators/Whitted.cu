#include <RayTracer/Integrators/Whitted.hpp>
#include <Camera/RayGeneratorWrapper.hpp>
#include <RayTracer/Film.hpp>
#include <Math/Geometry.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>
#include <Material/MaterialWrapper.hpp>
#include <Light/LightWrapper.hpp>
#include <Material/Material.hpp>
#include <RayTracer/Integrators/Utilities.hpp>

WhittedIntegrator::WhittedIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
    const uint32_t maxDepth, const uint32_t spp)
    : mSequenceGenerator(sequenceGenerator), mMaxDepth(maxDepth), mSpp(spp) {}

static DEVICE Spectrum Li(RenderingContext& context, const Ray& ray, uint32_t depth);

static DEVICE Spectrum specularReflect(RenderingContext& context, const Ray& ray,
    const Interaction& interaction, const Bsdf& bsdf, const uint32_t depth) {
    constexpr auto type = BxDFType::Reflection | BxDFType::Specular;
    const auto sampleF = bsdf.sampleF(Vector{interaction.wo}, context.sample(), type);

    auto&& shading = interaction.shadingGeometry;
    const auto idn = dot(sampleF.wi, Vector{shading.normal});

    if (sampleF.pdf > 0.0f & sampleF.f.lum() > 0.0f & idn != 0.0f) {
        auto newRay = interaction.spawnRay(Vector{sampleF.wi});
        newRay.xOri = interaction.pos + interaction.dpdx;
        newRay.yOri = interaction.pos + interaction.dpdy;
        {
            const Normal dndx{
                shading.dndu * interaction.duvdx.s +
                shading.dndv * interaction.duvdx.t
            };
            const Normal dndy{
                shading.dndu * interaction.duvdy.s +
                shading.dndv * interaction.duvdy.t
            };

            const auto dwodx = -ray.xDir - Vector{interaction.wo};
            const auto dwody = -ray.yDir - Vector{interaction.wo};

            const auto dDNdx = dot(dwodx, Vector{shading.normal}) + dot(interaction.wo, Vector{dndx});
            const auto dDNdy = dot(dwody, Vector{shading.normal}) + dot(interaction.wo, Vector{dndy});

            const auto k = dot(interaction.wo, Vector{shading.normal});

            newRay.xDir = Vector{sampleF.wi} - dwodx + Vector{(dndx * k + shading.normal * dDNdx)} * 2.0f;
            newRay.yDir = Vector{sampleF.wi} - dwody + Vector{(dndy * k + shading.normal * dDNdy)} * 2.0f;
        }

        return sampleF.f * Li(context, newRay, depth) * fabs(idn) / sampleF.pdf;
    }
    return Spectrum{};
}

static DEVICE Spectrum specularTransmit(RenderingContext& context, const Ray& ray,
    const Interaction& interaction, const Bsdf& bsdf, const uint32_t depth) {
    constexpr auto type = BxDFType::Transmission | BxDFType::Specular;
    const auto sampleF = bsdf.sampleF(Vector{interaction.wo}, context.sample(), type);

    auto&& shading = interaction.shadingGeometry;
    const auto idn = dot(sampleF.wi, Vector{shading.normal});

    if (sampleF.pdf > 0.0f & sampleF.f.lum() > 0.0f & idn != 0.0f) {
        auto newRay = interaction.spawnRay(Vector{sampleF.wi});
        newRay.xOri = interaction.pos + interaction.dpdx;
        newRay.yOri = interaction.pos + interaction.dpdy;
        {
            const auto odn = dot(interaction.wo, Vector{shading.normal});
            const auto eta = odn < 0.0f ? 1.0f / bsdf.getEta() : bsdf.getEta();

            const Normal dndx{
                shading.dndu * interaction.duvdx.s +
                shading.dndv * interaction.duvdx.t
            };

            const Normal dndy{
                shading.dndu * interaction.duvdy.s +
                shading.dndv * interaction.duvdy.t
            };

            const auto dwodx = -ray.xDir - Vector{interaction.wo};
            const auto dwody = -ray.yDir - Vector{interaction.wo};

            const auto dDNdx = dot(dwodx, Vector{shading.normal}) + dot(interaction.wo, Vector{dndx});
            const auto dDNdy = dot(dwody, Vector{shading.normal}) + dot(interaction.wo, Vector{dndy});

            const auto mu = eta * -odn - idn;
            const auto k = (eta - (eta * eta * -odn) / idn);
            const auto dmudx = k * dDNdx;
            const auto dmudy = k * dDNdy;

            newRay.xDir = Vector{sampleF.wi} + eta * dwodx - Vector{(dndx * mu + shading.normal * dmudx)};
            newRay.yDir = Vector{sampleF.wi} + eta * dwody - Vector{(dndy * mu + shading.normal * dmudy)};
        }

        return sampleF.f * Li(context, newRay, depth) * fabs(idn) / sampleF.pdf;
    }
    return Spectrum{};
}

static DEVICE Spectrum Li(RenderingContext& context, const Ray& ray, const uint32_t depth) {
    Interaction interaction;
    if (context.scene.intersect(ray, interaction)) {
        interaction.prepare(ray);
        Bsdf bsdf(interaction);
        interaction.material->computeScatteringFunctions(bsdf);
        auto L = interaction.le(Vector{interaction.wo});
        for (auto&& light : context.scene) {
            const auto sample = light->sampleLi(context.sample(), interaction.pos);
            if (sample.illumination.lum() > 0.0f & sample.pdf > 0.0f
                & !context.scene.intersect(interaction.spawnTo(sample.src))) {
                const auto f = bsdf.f(Vector{interaction.wo}, Vector{sample.wi});
                if (f.lum() > 0.0f)
                    L += f * sample.illumination
                        * (fabs(dot(sample.wi, Vector{interaction.shadingGeometry.normal})) / sample.pdf);
            }
        }
        if (depth > 1) {
            L += specularReflect(context, ray, interaction, bsdf, depth - 1);
            L += specularTransmit(context, ray, interaction, bsdf, depth - 1);
        }
        return L;
    }
    Spectrum L{};
    for (auto&& light : context.scene)
        L += light->le(ray);
    return L;
}

static GLOBAL void renderKernel(const RayGeneratorWrapper rayGenerator,
    const Transform toWorld, const vec2 offset,
    const SequenceGenerator2DWrapper sequenceGenerator, const uint32_t seqOffset,
    FilmTileRef dst, const uint32_t maxDepth, const SceneRef scene, const uint32_t spp,
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

void WhittedIntegrator::render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraToWorld,
    const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2 offset, const uvec2 dstSize) const {
    const uint32_t blockSize = DeviceMonitor::get().getProp().maxThreadsPerBlock;
    const auto size = filmTile.size();
    constexpr auto baseStack = sizeof(RayGeneratorWrapper) + sizeof(Transform) +
        sizeof(SequenceGenerator2DWrapper) + sizeof(SceneRef) + sizeof(RenderingContext) +
        sizeof(CameraSample) + sizeof(Spectrum) + 128U;
    constexpr auto liStack = sizeof(Interaction) + sizeof(Bsdf) + sizeof(Ray) + sizeof(BxDFSample);
    const auto stackSize = baseStack + liStack * (mMaxDepth + 3);
    const auto maxLaunchSize = std::min(blockSize, static_cast<uint32_t>(
        (DeviceMonitor::get().getMemoryFreeSize() * 4 / 5) / (stackSize * size.x * size.y)));
    auto todo = mSpp;
    while (todo) {
        const auto current = std::min(todo, maxLaunchSize);
        todo -= current;
        buffer.launchKernelDim(makeKernelDesc(renderKernel, stackSize),
            dim3{size.x, size.y, calcBlockSize(current, blockSize)},
            dim3{std::min(blockSize, current)}, rayGenerator, cameraToWorld, static_cast<vec2>(offset),
            mSequenceGenerator, todo * (offset.x << 16 | offset.y), filmTile.toRef(), mMaxDepth,
            scene.toRef(), current, 1.0f / vec2(dstSize));
    }
}
