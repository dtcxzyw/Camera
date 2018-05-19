#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Metal final : Material {
private:
    Texture2DSpectrum mK, mEta;
    Texture2DFloat mRoughnessX, mRoughnessY;
public:
    explicit Metal(const Texture2DSpectrum& k, const Texture2DSpectrum& eta,
        const Texture2DFloat& roughnessX, const Texture2DFloat& roughnessY)
        : mK(k), mEta(eta), mRoughnessX(roughnessX), mRoughnessY(roughnessY) {}

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, TransportMode) const {
        const FresnelWrapper fresnel{
            FresnelConductor{
                Spectrum{1.0f},
                mEta.sample(bsdf.getInteraction()), mK.sample(bsdf.getInteraction())
            }
        };
        const MicrofactDistributionWrapper distribution{
            TrowbridgeReitzDistribution{
                mRoughnessX.sample(bsdf.getInteraction()),
                mRoughnessY.sample(bsdf.getInteraction())
            }
        };
        bsdf.add(MicrofacetReflection{Spectrum{1.0f}, fresnel, distribution});
    }
};
