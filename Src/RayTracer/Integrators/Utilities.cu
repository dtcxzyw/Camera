#include <RayTracer/Integrators/Utilities.hpp>
#include <Light/LightWrapper.hpp>
#include <Math/Interaction.hpp>

DEVICEINLINE float powerHeuristic(const float f, const float g) {
    const auto f2 = f * f, g2 = g * g;
    return f2 / (f2 + g2);
}

DEVICE Spectrum estimateDirect(RenderingContext& context, const Interaction& interaction,
    const Bsdf& bsdf, const LightWrapper& light) {
    Spectrum L{};
    const auto sample = light.sampleLi(context.sample(), interaction.pos);
    const auto absDot = fabs(dot(sample.wi, Vector{ interaction.shadingGeometry.normal }));
    //sample light
    if (sample.illumination.lum() > 0.0f & sample.pdf > 0.0f) {
        const auto f = bsdf.f(Vector{interaction.wo}, Vector{sample.wi}) * sample.illumination;
        auto fac = absDot / sample.pdf;
        if ((f.lum() > 0.0f & fac > 0.0f) && !context.scene.intersect(interaction.spawnTo(sample.src))) {
            if (!light.isDelta()) {
                const auto pdf = bsdf.pdf(Vector{interaction.wo}, Vector{sample.wi});
                fac *= powerHeuristic(sample.pdf, pdf);
            }
            L = f * fac;
        }
    }
    //sample bsdf
    //TODO:need AreaLight
    /*
    if (!light.isDelta()) {
        auto sample = bsdf.sampleF(Vector{ interaction.wo }, context.sample());
        sample.f *= absDot;
        const auto specular = static_cast<bool>(sample.type&BxDFType::Specular);
        auto weight = 1.0f;
        if (!specular) {
            const auto pdf = light.pdfLi();
            if (pdf == 0.0f)return L;
            
        }
    }
    */
    return L;
}

DEVICE Spectrum uniformSampleOneLight(RenderingContext& context, const Interaction& interaction,
    const Bsdf& bsdf) {
    const auto size = context.scene.size();
    if (size == 0)return Spectrum{};
    const auto id = min(static_cast<unsigned int>(context.sample().x * size), size - 1);
    return size * estimateDirect(context, interaction, bsdf, **(context.scene.begin() + id));
}
