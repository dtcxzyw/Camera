#pragma once
#include <Texture/TextureMapping.hpp>
#include <Texture/TextureMapping2DWarpper.hpp>
#include <Texture/TextureSampler2DFloatWarpper.hpp>
#include <Texture/TextureSampler2DSpectrumWarpper.hpp>
#include <Texture/TextureSampler3DFloatWarpper.hpp>
#include <Texture/TextureSampler3DSpectrumWarpper.hpp>

//TODO:mix texture
class Texture2DFloat final {
private:
    TextureMapping2DWarpper mMapping;
    TextureSampler2DFloatWarpper mSampler;
public:
    explicit Texture2DFloat(const TextureMapping2DWarpper& mapping,
        const TextureSampler2DFloatWarpper& sampler) :mMapping(mapping), mSampler(sampler) {}
    DEVICE float sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};

class Texture2DSpectrum final {
private:
    TextureMapping2DWarpper mMapping;
    TextureSampler2DSpectrumWarpper mSampler;
public:
    explicit Texture2DSpectrum(const TextureMapping2DWarpper& mapping,
        const TextureSampler2DSpectrumWarpper& sampler)
    :mMapping(mapping), mSampler(sampler) {}
    DEVICE Spectrum sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};

class Texture3DFloat final {
private:
    TextureMapping3D mMapping;
    TextureSampler3DFloatWarpper mSampler;
public:
    explicit Texture3DFloat(const TextureMapping3D& mapping,
        const TextureSampler3DFloatWarpper& sampler) :mMapping(mapping), mSampler(sampler) {}
    DEVICE float sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};

class Texture3DSpectrum final {
private:
    TextureMapping3D mMapping;
    TextureSampler3DSpectrumWarpper mSampler;
public:
    explicit Texture3DSpectrum(const TextureMapping3D& mapping,
        const TextureSampler3DSpectrumWarpper& sampler)
        :mMapping(mapping), mSampler(sampler) {}
    DEVICE Spectrum sample(const Interaction& interaction) const {
        return mSampler.sample(mMapping.map(interaction));
    }
};
