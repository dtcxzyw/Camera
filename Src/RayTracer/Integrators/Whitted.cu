#include <RayTracer/Integrators/Whitted.hpp>
#include <RayTracer/Scene.hpp>
#include <Camera/RayGeneratorWrapper.hpp>
#include <RayTracer/Film.hpp>
#include <Math/Geometry.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>
#include <Material/MaterialWrapper.hpp>
#include <Light/LightWrapper.hpp>
#include <Material/Material.hpp>

WhittedIntegrator::WhittedIntegrator(const SequenceGenerator2DWrapper& sequenceGenerator,
    const unsigned int maxDepth, const unsigned int spp)
    : mSequenceGenerator(sequenceGenerator), mMaxDepth(maxDepth), mSpp(spp) {}

struct RenderingContext {
    const SequenceGenerator2DWrapper& sequenceGenerator;
    const SceneRef& scene;
    unsigned int offset;

    DEVICE RenderingContext(const SequenceGenerator2DWrapper& sequenceGenerator,
        const SceneRef& scene, const unsigned int offset)
        : sequenceGenerator(sequenceGenerator), scene(scene), offset(offset) {}

    DEVICE vec2 sample() {
        return sequenceGenerator.sample(++offset);
    }
};

DEVICE Spectrum Li(RenderingContext& context, const Ray& ray, unsigned int depth);

DEVICE Spectrum specularReflect(RenderingContext& context, const Ray& ray,
    const Interaction& interaction, const Bsdf& bsdf, const unsigned int depth) {
    constexpr auto type = BxDFType::Reflection | BxDFType::Specular;
    const auto sampleF = bsdf.sampleF(Vector{interaction.wo}, context.sample(), type);

    const auto idn = dot(sampleF.wi, interaction.normal);

    if (sampleF.pdf > 0.0f & sampleF.f.lum() > 0.0f & idn != 0.0f) {
        auto newRay = interaction.spawnRay(Vector{sampleF.wi});
        newRay.xOri = interaction.pos + interaction.dpdx;
        newRay.yOri = interaction.pos + interaction.dpdy;

        const Normal dndx{
            interaction.dndu * interaction.duvdx.s +
            interaction.dndv * interaction.duvdx.t
        };
        const Normal dndy{
            interaction.dndu * interaction.duvdy.s +
            interaction.dndv * interaction.duvdy.t
        };

        const auto dwodx = -ray.xDir - Vector{interaction.wo};
        const auto dwody = -ray.yDir - Vector{interaction.wo};

        const auto dDNdx = dot(dwodx, Vector{interaction.normal}) + dot(interaction.wo, dndx);
        const auto dDNdy = dot(dwody, Vector{interaction.normal}) + dot(interaction.wo, dndy);

        const auto k = dot(interaction.wo, interaction.normal);

        newRay.xDir = Vector{sampleF.wi} - dwodx + 2.0f * (dndx * k + interaction.normal * dDNdx);
        newRay.yDir = Vector{sampleF.wi} - dwody + 2.0f * (dndy * k + interaction.normal * dDNdy);

        return sampleF.f * Li(context, newRay, depth) * fabs(idn) / sampleF.pdf;
    }
    return Spectrum{0.0f};
}

DEVICE Spectrum specularTransmit(RenderingContext& context, const Ray& ray,
    const Interaction& interaction, const Bsdf& bsdf, const unsigned int depth) {
    constexpr auto type = BxDFType::Transmission | BxDFType::Specular;
    const auto sampleF = bsdf.sampleF(Vector{interaction.wo}, context.sample(), type);

    const auto idn = dot(sampleF.wi, interaction.normal);

    if (sampleF.pdf > 0.0f & sampleF.f.lum() > 0.0f & idn != 0.0f) {
        auto newRay = interaction.spawnRay(Vector{sampleF.wi});
        newRay.xOri = interaction.pos + interaction.dpdx;
        newRay.yOri = interaction.pos + interaction.dpdy;

        const auto eta = dot(interaction.wo, interaction.normal) < 0.0f ? 1.0f / bsdf.getEta() : bsdf.getEta();

        const Normal dndx{
            interaction.dndu * interaction.duvdx.s +
            interaction.dndv * interaction.duvdx.t
        };

        const Normal dndy{
            interaction.dndu * interaction.duvdy.s +
            interaction.dndv * interaction.duvdy.t
        };

        const auto dwodx = -ray.xDir - Vector{interaction.wo};
        const auto dwody = -ray.yDir - Vector{interaction.wo};

        const auto dDNdx = dot(dwodx, Vector{interaction.normal}) + dot(interaction.wo, dndx);
        const auto dDNdy = dot(dwody, Vector{interaction.normal}) + dot(interaction.wo, dndy);

        const auto odn = dot(interaction.wo, interaction.normal);
        const auto mu = eta * -odn - idn;
        const auto k = (eta - (eta * eta * -odn) / idn);
        const auto dmudx = k * dDNdx;
        const auto dmudy = k * dDNdy;

        newRay.xDir = Vector{sampleF.wi} + eta * dwodx - (dndx * mu + interaction.normal * dmudx);
        newRay.yDir = Vector{sampleF.wi} + eta * dwody - (dndy * mu + interaction.normal * dmudy);

        return sampleF.f * Li(context, newRay, depth) * fabs(idn) / sampleF.pdf;
    }
    return Spectrum{0.0f};
}

DEVICE Spectrum Li(RenderingContext& context, const Ray& ray, const unsigned int depth) {
    Interaction interaction;
    if (context.scene.intersect(ray, interaction)) {
        printf("pixel C\n");
        return Spectrum{};
        interaction.prepare(ray);
        Bsdf bsdf(interaction);
        interaction.material->computeScatteringFunctions(bsdf);
        Spectrum L{};
        for (auto&& light : context.scene) {
            const auto sample = light->sampleLi(context.sample(), interaction.pos);
            if (sample.illumination.lum() > 0.0f & sample.pdf > 0.0f
                & !context.scene.intersect(interaction.spawnTo(sample.src))) {
                const auto f = bsdf.f(Vector{interaction.wo}, Vector{sample.wi});
                if (f.lum() > 0.0f)
                    L += f * sample.illumination
                        * fabs(dot(sample.wi, interaction.normal)) / sample.pdf;
            }
        }
        if (depth > 1) {
            L += specularReflect(context, ray, interaction, bsdf, depth - 1);
            L += specularTransmit(context, ray, interaction, bsdf, depth - 1);
        }
        return L;
    }
    printf("pixel D\n");
    return Spectrum{};
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
    if (weight > 0.0f) {
        const auto L = Li(context, toWorld(ray), maxDepth);
        return;
        //bug
        dst.add(tilePos, L * weight);
    }
}

void WhittedIntegrator::render(CommandBuffer& buffer, const SceneDesc& scene, const Transform& cameraTransform,
    const RayGeneratorWrapper& rayGenerator, FilmTile& filmTile, const uvec2& offset) const {
    const unsigned int blockSize = DeviceMonitor::get().getProp().maxThreadsPerBlock;
    const auto size = filmTile.size();
    buffer.launchKernelDim(renderKernel, dim3{size.x, size.y, calcBlockSize(mSpp, blockSize)},
        dim3{ std::min(blockSize,mSpp) }, rayGenerator, cameraTransform, static_cast<vec2>(offset),
        mSequenceGenerator, offset.x << 16 | offset.y, filmTile.toRef(), mMaxDepth, scene.toRef());
}
