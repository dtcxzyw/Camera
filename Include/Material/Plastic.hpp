#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Plastic final :Material {
private:
    Texture2DSpectrum mKd;
    Texture2DSpectrum mKs;
    Texture2DFloat mRoughness;
public:
    explicit Plastic(const Texture2DSpectrum& kd, const Texture2DSpectrum& ks,
        const Texture2DFloat& roughness) :mKd(kd), mKs(ks), mRoughness(roughness) {}
    DEVICE void computeScatteringFunctions(Bsdf& bsdf, TransportMode mode) {
        //diffuse
        const auto kd = mKd.sample(bsdf.getInteraction());
        bsdf.add(BxDFWarpper{ LambertianReflection{kd} });

        //specular
        const auto ks = mKs.sample(bsdf.getInteraction());
        if (!ks.IsBlack()) {
            Fresnel *fresnel = ARENA_ALLOC(arena, FresnelDielectric)(1.5f, 1.f);
            // Create microfacet distribution _distrib_ for plastic material
            Float rough = roughness->Evaluate(*si);
            if (remapRoughness)
                rough = TrowbridgeReitzDistribution::RoughnessToAlpha(rough);
            MicrofacetDistribution *distrib =
                ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(rough, rough);
            BxDF *spec =
                ARENA_ALLOC(arena, MicrofacetReflection)(ks, distrib, fresnel);
            si->bsdf->Add(spec);
        }
    };
}
