#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Subtrate final : Material {
private:
    Texture2DSpectrum mKd;
    Texture2DSpectrum mKs;
    Texture2DFloat mRoughnessX, mRoughnessY;
public:
    explicit Subtrate(const Texture2DSpectrum& kd, const Texture2DSpectrum& ks,
        const Texture2DFloat& roughnessX, const Texture2DFloat& roughnessY)
        : mKd(kd), mKs(ks), mRoughnessX(roughnessX), mRoughnessY(roughnessY) {}

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, TransportMode) const {
        const auto kd = mKd.sample(bsdf.getInteraction());
        const auto ks = mKs.sample(bsdf.getInteraction());
        if (kd.y() > 0.0f | ks.y() > 0.0f) {
            const auto roughnessX = mRoughnessX.sample(bsdf.getInteraction());
            const auto roughnessY = mRoughnessY.sample(bsdf.getInteraction());
            const MicrofactDistributionWrapper distribution{
                TrowbridgeReitzDistribution{roughnessX, roughnessY}
            };
            bsdf.add(FresnelBlend{kd, ks, distribution});
        }
    }
};
