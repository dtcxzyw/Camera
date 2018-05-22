#pragma once
#include <Texture/TextureMapping.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler2DSpectrumWrapper final {
private:
    enum class TextureSampler2DSpectrumClassType {
        ConstantSampler2DSpectrum,
        TextureMapSpectrumSampler,
        UVSampler,
        CheckBoardSampler,
        PolkaDotsSampler
    };

    union {
        char unused{};
        ConstantSampler2DSpectrum dataConstantSampler2DSpectrum;
        TextureMapSpectrumSampler dataTextureMapSpectrumSampler;
        UVSampler dataUVSampler;
        CheckBoardSampler dataCheckBoardSampler;
        PolkaDotsSampler dataPolkaDotsSampler;
    };

    TextureSampler2DSpectrumClassType mType;
public:
    TextureSampler2DSpectrumWrapper(): mType(static_cast<TextureSampler2DSpectrumClassType>(15)) {};

    explicit TextureSampler2DSpectrumWrapper(const ConstantSampler2DSpectrum& data)
        : dataConstantSampler2DSpectrum(data), mType(TextureSampler2DSpectrumClassType::ConstantSampler2DSpectrum) {}

    explicit TextureSampler2DSpectrumWrapper(const TextureMapSpectrumSampler& data)
        : dataTextureMapSpectrumSampler(data), mType(TextureSampler2DSpectrumClassType::TextureMapSpectrumSampler) {}

    explicit TextureSampler2DSpectrumWrapper(const UVSampler& data)
        : dataUVSampler(data), mType(TextureSampler2DSpectrumClassType::UVSampler) {}

    explicit TextureSampler2DSpectrumWrapper(const CheckBoardSampler& data)
        : dataCheckBoardSampler(data), mType(TextureSampler2DSpectrumClassType::CheckBoardSampler) {}

    explicit TextureSampler2DSpectrumWrapper(const PolkaDotsSampler& data)
        : dataPolkaDotsSampler(data), mType(TextureSampler2DSpectrumClassType::PolkaDotsSampler) {}

    TextureSampler2DSpectrumWrapper(const TextureSampler2DSpectrumWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureSampler2DSpectrumWrapper));
    }

    TextureSampler2DSpectrumWrapper& operator=(const TextureSampler2DSpectrumWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureSampler2DSpectrumWrapper));
    return *this;
    }

    DEVICE Spectrum sample(const TextureMapping2DInfo& info) const {
        switch (mType) {
            case TextureSampler2DSpectrumClassType::ConstantSampler2DSpectrum: return dataConstantSampler2DSpectrum.sample(info);
            case TextureSampler2DSpectrumClassType::TextureMapSpectrumSampler: return dataTextureMapSpectrumSampler.sample(info);
            case TextureSampler2DSpectrumClassType::UVSampler: return dataUVSampler.sample(info);
            case TextureSampler2DSpectrumClassType::CheckBoardSampler: return dataCheckBoardSampler.sample(info);
            case TextureSampler2DSpectrumClassType::PolkaDotsSampler: return dataPolkaDotsSampler.sample(info);
        }
    }

};
