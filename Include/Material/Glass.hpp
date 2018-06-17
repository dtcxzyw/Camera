#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Glass final : Material {
private:
    Texture2DSpectrum mKr, mKt;
    Texture2DFloat mIndex, mRoughnessX, mRoughnessY;
public:
    explicit Glass(const Texture2DSpectrum& kr, const Texture2DSpectrum& kt,
        const Texture2DFloat& index, const Texture2DFloat& roughnessX,
        const Texture2DFloat& roughnessY)
        : mKr(kr), mKt(kt), mIndex(index), mRoughnessX(roughnessX), mRoughnessY(roughnessY) {}

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, const TransportMode mode) const {
        const auto eta = mIndex.sample(bsdf.getInteraction());
        bsdf.setEta(eta);
        const auto roughnessX = mRoughnessX.sample(bsdf.getInteraction());
        const auto roughnessY = mRoughnessY.sample(bsdf.getInteraction());
        const auto isSpecular = roughnessX == 0.0f & roughnessY == 0.0f;
        const auto kr = mKr.sample(bsdf.getInteraction());
        const auto kt = mKt.sample(bsdf.getInteraction());
        if (isSpecular) {
            if (kr.y() > 0.0f | kt.y() > 0.0f)
                bsdf.add(FresnelSpecular(kr, kt, 1.0f, eta, mode));
        }
        else {
            const MicrofactDistributionWrapper distribution{
                TrowbridgeReitzDistribution{roughnessX, roughnessY}
            };
            if (kr.y() > 0.0f) {
                const FresnelWrapper fresnel{FresnelDielectric{1.0f, eta}};
                bsdf.add(MicrofacetReflection{kr, fresnel, distribution});
            }
            if (kt.y() > 0.0f) {
                bsdf.add(MicrofacetTransmission{kt, 1.0f, eta, distribution, mode});
            }
        }
    }
};
