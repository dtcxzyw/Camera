#pragma once
#include <Math/Interaction.hpp>
#include <Light/LightWrapper.hpp>

Spectrum SurfaceInteraction::le(const Vector& w) const {
    return areaLight ? areaLight->emitL(*this, w) : Spectrum{};
}
