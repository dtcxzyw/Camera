#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Matte final : Material {
private:
    Texture2DSpectrum mKd;
    Texture2DFloat mSigma;
public:
    explicit Matte(const Texture2DSpectrum& kd, const Texture2DFloat& sigma)
        : mKd(kd), mSigma(sigma) {}

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, TransportMode) const {
        const auto kd = mKd.sample(bsdf.getInteraction());
        if (kd.lum() > 0.0f) {
            const auto sigma = mSigma.sample(bsdf.getInteraction());
            if (sigma)bsdf.add(OrenNayar{kd, sigma});
            else bsdf.add(LambertianReflection{kd});
        }
    }
};
