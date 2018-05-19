#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Plastic final : Material {
private:
    Texture2DSpectrum mKd;
    Texture2DSpectrum mKs;
    Texture2DFloat mRoughness;
public:
    explicit Plastic(const Texture2DSpectrum& kd, const Texture2DSpectrum& ks,
        const Texture2DFloat& roughness) : mKd(kd), mKs(ks), mRoughness(roughness) {}

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, TransportMode) const {
        //diffuse
        const auto kd = mKd.sample(bsdf.getInteraction());
        bsdf.add(LambertianReflection{kd});

        //specular
        const auto ks = mKs.sample(bsdf.getInteraction());
        if (ks.lum() > 0.0f) {
            const FresnelWrapper fresnel{FresnelDielectric{1.5f, 1.0f}};
            const auto roughness = mRoughness.sample(bsdf.getInteraction());
            const MicrofactDistributionWrapper distribution{
                TrowbridgeReitzDistribution{roughness, roughness}
            };
            bsdf.add(MicrofacetReflection{ ks, fresnel, distribution });
        }
    }
};
