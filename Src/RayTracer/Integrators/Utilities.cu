#include <RayTracer/Integrators/Utilities.hpp>
#include <Light/LightWrapper.hpp>
#include <Math/Interaction.hpp>
#include <Light/LightDistribution.hpp>

DEVICEINLINE float powerHeuristic(const float f, const float g) {
    const auto f2 = f * f, g2 = g * g;
    return f2 / (f2 + g2);
}

DEVICE Spectrum estimateDirect(RenderingContext& context, const SurfaceInteraction& interaction,
    const Bsdf& bsdf, const LightWrapper& light) {
    Spectrum L{};
    //sample light
    {
        const auto sample = light.sampleLi(context.sample(), interaction);
        if (sample.illumination.y() > 0.0f & sample.pdf > 0.0f) {
            const auto f = bsdf.f(Vector{ interaction.wo }, Vector{ sample.wi }) * sample.illumination;
            auto fac = fabs(dot(sample.wi, Vector{ interaction.shadingGeometry.normal })) / sample.pdf;
            if ((f.y() > 0.0f & fac > 0.0f) && !context.scene.intersect(interaction.spawnTo(sample.src))) {
                if (!light.isDelta()) {
                    const auto pdf = bsdf.pdf(Vector{ interaction.wo }, Vector{ sample.wi });
                    fac *= powerHeuristic(sample.pdf, pdf);
                }
                L = f * fac;
            }
        }
    }
    //sample bsdf
    if (!light.isDelta()) {
        auto sampleF = bsdf.sampleF(Vector{ interaction.wo }, context.sample());
        sampleF.f *= fabs(dot(sampleF.wi, Vector{ interaction.shadingGeometry.normal }));
        const auto specular = static_cast<bool>(sampleF.type&BxDFType::Specular);
        if (sampleF.f.y() > 0.0f & sampleF.pdf > 0.0f) {
            auto weight = 1.0f;
            if (!specular) {
                const auto lightPdf = light.pdfLi(interaction, sampleF.wi);
                if (lightPdf == 0.0f) return L;
                weight = powerHeuristic(sampleF.pdf, lightPdf);
            }

            SurfaceInteraction lightIsect;
            const auto ray = interaction.spawnRay(sampleF.wi);
            const auto res = context.scene.intersect(ray, lightIsect);

            Spectrum Li(0.f);
            if (res) {
                if (lightIsect.areaLight == &light)
                    Li = lightIsect.le(-sampleF.wi);
            }
            else Li = light.le(ray);
            if (Li.y() > 0.0f) L += sampleF.f * Li * (weight / sampleF.pdf);
        }
    }
    CHECKFP(L.y());
    return L;
}

DEVICE Spectrum uniformSampleOneLight(RenderingContext& context, const SurfaceInteraction& interaction,
    const Bsdf& bsdf, const LightDistribution* distribution) {
    const auto size = context.scene.size();
    if (size == 0)return Spectrum{};
    unsigned int id;
    float invPdf;
    const auto sample = context.sample().x;
    if (distribution) {
        float pdf;
        id = distribution->chooseOneLight(sample, pdf);
        if (pdf <= 0.0f)return Spectrum{};
        invPdf = 1.0f / pdf;
    }
    else id = min(static_cast<uint32_t>(sample * size), size - 1), invPdf = size;
    return estimateDirect(context, interaction, bsdf, context.scene[id]) * invPdf;
}
