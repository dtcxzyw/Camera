#pragma once
#include <Material/Material.hpp>
#include <Texture/Texture.hpp>

class Plastic final :public Material {
private:
    Texture2DSpectrum mKd;
    Texture2DSpectrum mKs;
    Texture2DFloat mRoughness;
public:

    MemorySpan<MaterialRef> toRef(CommandBuffer& buffer) const override;
};
