#pragma once
#include <Texture/TextureMapping.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler2DSpectrumWarpper final {
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
    TextureSampler2DSpectrumWarpper(): mType(static_cast<TextureSampler2DSpectrumClassType>(15)) {};

    explicit TextureSampler2DSpectrumWarpper(const ConstantSampler2DSpectrum& data)
        : dataConstantSampler2DSpectrum(data), mType(TextureSampler2DSpectrumClassType::ConstantSampler2DSpectrum) {}

    explicit TextureSampler2DSpectrumWarpper(const TextureMapSpectrumSampler& data)
        : dataTextureMapSpectrumSampler(data), mType(TextureSampler2DSpectrumClassType::TextureMapSpectrumSampler) {}

    explicit TextureSampler2DSpectrumWarpper(const UVSampler& data)
        : dataUVSampler(data), mType(TextureSampler2DSpectrumClassType::UVSampler) {}

    explicit TextureSampler2DSpectrumWarpper(const CheckBoardSampler& data)
        : dataCheckBoardSampler(data), mType(TextureSampler2DSpectrumClassType::CheckBoardSampler) {}

    explicit TextureSampler2DSpectrumWarpper(const PolkaDotsSampler& data)
        : dataPolkaDotsSampler(data), mType(TextureSampler2DSpectrumClassType::PolkaDotsSampler) {}

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
