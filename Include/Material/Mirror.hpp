#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Mirror final : Material {
private:
    Texture2DSpectrum mR;
public:
    explicit Mirror(const Texture2DSpectrum& r) : mR(r) {}

    DEVICE void computeScatteringFunctions(Bsdf& bsdf, TransportMode) const {
        const auto reflection = mR.sample(bsdf.getInteraction());
        if (reflection.lum() > 0.0f)
            bsdf.add(SpecularReflection{reflection, FresnelWrapper{FresnelNoOp{}}});
    }
};
