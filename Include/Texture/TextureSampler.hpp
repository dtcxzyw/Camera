#pragma once
#include <Core/Builtin.hpp>
#include <Texture/TextureMapping.hpp>
#include <Texture/ProceduralTexture.hpp>
#include <Texture/Noise.hpp>

struct TextureSampler2DFloatTag {};

struct TextureSampler2DSpectrumTag {};

class ConstantSampler2DFloat final : TextureSampler2DFloatTag {
private:
    float mValue;
public:
    explicit ConstantSampler2DFloat(const float val) : mValue(val) {}
    DEVICE float sample(const TextureMapping2DInfo&) const {
        return mValue;
    }
};

class ConstantSampler2DSpectrum final : TextureSampler2DSpectrumTag {
private:
    Spectrum mValue;
public:
    explicit ConstantSampler2DSpectrum(const Spectrum val) : mValue(val) {}
    DEVICE Spectrum sample(const TextureMapping2DInfo&) const {
        return mValue;
    }
};

class TextureMapFloatSampler final : TextureSampler2DFloatTag {
private:
    BuiltinSamplerRef<float> mSampler;
public:
    explicit TextureMapFloatSampler(const BuiltinSampler<float>& sampler)
        : mSampler(sampler.toRef()) {}

    DEVICE float sample(const TextureMapping2DInfo& info) const {
        return mSampler.getGrad(info.pos, info.dpdx, info.dpdy);
    }
};

class TextureMapSpectrumSampler final : TextureSampler2DSpectrumTag {
private:
    BuiltinSamplerRef<RGBA> mSampler;
public:
    explicit TextureMapSpectrumSampler(const BuiltinSampler<RGBA>& sampler)
        : mSampler(sampler.toRef()) {}

    DEVICE Spectrum sample(const TextureMapping2DInfo& info) const {
        return Spectrum{mSampler.getGrad(info.pos, info.dpdx, info.dpdy)};
    }
};

class UVSampler final : public TextureSampler2DSpectrumTag {
public:
    DEVICE Spectrum sample(const TextureMapping2DInfo& info) const {
        return uv(info.pos);
    }
};

class CheckBoardSampler final : TextureSampler2DSpectrumTag {
private:
    Spectrum mA, mB;
public:
    explicit CheckBoardSampler(const Spectrum& a = Spectrum{0.0f}, const Spectrum& b = Spectrum{1.0f})
        : mA(a), mB(b) {}

    DEVICE Spectrum sample(const TextureMapping2DInfo& info) const {
        return checkerBoard(info.pos, mA, mB);
    }
};

class PolkaDotsSampler final : TextureSampler2DSpectrumTag {
private:
    Spectrum mA, mB;
    const float mRadius;
public:
    explicit PolkaDotsSampler(const float radius, const Spectrum& a = Spectrum{0.0f},
        const Spectrum& b = Spectrum{1.0f})
        : mA(a), mB(b), mRadius(radius) {}

    DEVICE Spectrum sample(const TextureMapping2DInfo& info) const {
        return polkaDots(info.pos, mRadius) ? mA : mB;
    }
};

struct TextureSampler3DFloatTag {};

struct TextureSampler3DSpectrumTag {};

class ConstantSampler3DFloat final : TextureSampler3DFloatTag {
private:
    float mValue;
public:
    explicit ConstantSampler3DFloat(const float val) : mValue(val) {}
    DEVICE float sample(const TextureMapping3DInfo&) const {
        return mValue;
    }
};

class ConstantSampler3DSpectrum final : TextureSampler3DSpectrumTag {
private:
    Spectrum mValue;
public:
    explicit ConstantSampler3DSpectrum(const Spectrum val) : mValue(val) {}
    DEVICE Spectrum sample(const TextureMapping3DInfo&) const {
        return mValue;
    }
};

class CheckBoard3DSampler final : TextureSampler3DSpectrumTag {
private:
    Spectrum mA, mB;
public:
    explicit CheckBoard3DSampler(const Spectrum& a = Spectrum{0.0f}, const Spectrum& b = Spectrum{1.0f})
        : mA(a), mB(b) {}

    DEVICE Spectrum sample(const TextureMapping3DInfo& info) const {
        return checkerBoard3D(info.pos, mA, mB);
    }
};

class FbmSampler final : TextureSampler3DFloatTag {
private:
    float mOmega;
    int mMaxOctaves;
public:
    explicit FbmSampler(const float omega, const int maxOctaves)
        : mOmega(omega), mMaxOctaves(maxOctaves) {}

    DEVICE float sample(const TextureMapping3DInfo& info) const {
        return fbm(info.pos, mOmega,
            fmin(mMaxOctaves, calcOctavesAntiAliased(info.dpdx, info.dpdy)));
    }
};

class TurbulenceSampler final : TextureSampler3DFloatTag {
private:
    float mOmega;
    int mMaxOctaves;
public:
    explicit TurbulenceSampler(const float omega, const int maxOctaves)
        : mOmega(omega), mMaxOctaves(maxOctaves) {}

    DEVICE float sample(const TextureMapping3DInfo& info) const {
        return turbulence(info.pos, mOmega,
            fmin(mMaxOctaves, calcOctavesAntiAliased(info.dpdx, info.dpdy)), mMaxOctaves);
    }
};

class WindyWavesSampler final : TextureSampler3DFloatTag {
public:
    DEVICE float sample(const TextureMapping3DInfo& info) const {
        return windyWaves(info.pos, calcOctavesAntiAliased(info.dpdx, info.dpdy));
    }
};

class MarbleSampler final : TextureSampler3DSpectrumTag {
private:
    float mVar;
    float mOmega;
public:
    explicit MarbleSampler(const float var, const float omega) : mVar(var), mOmega(omega) {}

    DEVICE Spectrum sample(const TextureMapping3DInfo& info) const {
        return Spectrum{marble(info.pos, mVar, mOmega, calcOctavesAntiAliased(info.dpdx, info.dpdy))};
    }
};
