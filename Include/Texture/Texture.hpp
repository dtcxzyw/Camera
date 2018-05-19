#pragma once
#include <Texture/TextureMapping.hpp>
#include <Texture/TextureMapping2DWrapper.hpp>
#include <Texture/TextureSampler2DFloatWrapper.hpp>
#include <Texture/TextureSampler2DSpectrumWrapper.hpp>
#include <Texture/TextureSampler3DFloatWrapper.hpp>
#include <Texture/TextureSampler3DSpectrumWrapper.hpp>

//TODO:mix texture
class Texture2DFloat final {
private:
    TextureMapping2DWrapper mMapping;
    TextureSampler2DFloatWrapper mSampler;
public:
    explicit Texture2DFloat(const TextureMapping2DWrapper& mapping,
        const TextureSampler2DFloatWrapper& sampler) :mMapping(mapping), mSampler(sampler) {}
    DEVICE float sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};

class Texture2DSpectrum final {
private:
    TextureMapping2DWrapper mMapping;
    TextureSampler2DSpectrumWrapper mSampler;
public:
    explicit Texture2DSpectrum(const TextureMapping2DWrapper& mapping,
        const TextureSampler2DSpectrumWrapper& sampler)
    :mMapping(mapping), mSampler(sampler) {}
    DEVICE Spectrum sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};

class Texture3DFloat final {
private:
    TextureMapping3D mMapping;
    TextureSampler3DFloatWrapper mSampler;
public:
    explicit Texture3DFloat(const TextureMapping3D& mapping,
        const TextureSampler3DFloatWrapper& sampler) :mMapping(mapping), mSampler(sampler) {}
    DEVICE float sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};

class Texture3DSpectrum final {
private:
    TextureMapping3D mMapping;
    TextureSampler3DSpectrumWrapper mSampler;
public:
    explicit Texture3DSpectrum(const TextureMapping3D& mapping,
        const TextureSampler3DSpectrumWrapper& sampler)
        :mMapping(mapping), mSampler(sampler) {}
    DEVICE Spectrum sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};
