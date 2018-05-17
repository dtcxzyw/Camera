#pragma once
#include <Texture/TextureMapping.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler3DSpectrumWarpper final {
private:
    enum class TextureSampler3DSpectrumClassType {
        ConstantSampler3DSpectrum,
        CheckBoard3DSampler,
        MarbleSampler
    };

    union {
        char unused{};
        ConstantSampler3DSpectrum dataConstantSampler3DSpectrum;
        CheckBoard3DSampler dataCheckBoard3DSampler;
        MarbleSampler dataMarbleSampler;
    };

    TextureSampler3DSpectrumClassType mType;
public:
    TextureSampler3DSpectrumWarpper(): mType(static_cast<TextureSampler3DSpectrumClassType>(15)) {};

    explicit TextureSampler3DSpectrumWarpper(const ConstantSampler3DSpectrum& data)
        : dataConstantSampler3DSpectrum(data), mType(TextureSampler3DSpectrumClassType::ConstantSampler3DSpectrum) {}

    explicit TextureSampler3DSpectrumWarpper(const CheckBoard3DSampler& data)
        : dataCheckBoard3DSampler(data), mType(TextureSampler3DSpectrumClassType::CheckBoard3DSampler) {}

    explicit TextureSampler3DSpectrumWarpper(const MarbleSampler& data)
        : dataMarbleSampler(data), mType(TextureSampler3DSpectrumClassType::MarbleSampler) {}

    DEVICE Spectrum sample(const TextureMapping3DInfo& info) const {
        switch (mType) {
            case TextureSampler3DSpectrumClassType::ConstantSampler3DSpectrum: return dataConstantSampler3DSpectrum.sample(info);
            case TextureSampler3DSpectrumClassType::CheckBoard3DSampler: return dataCheckBoard3DSampler.sample(info);
            case TextureSampler3DSpectrumClassType::MarbleSampler: return dataMarbleSampler.sample(info);
        }
    }

};
