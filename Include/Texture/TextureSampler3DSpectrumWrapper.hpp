#pragma once
#include <Texture/TextureMapping.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler3DSpectrumWrapper final {
private:
    enum class TextureSampler3DSpectrumClassType {
        ConstantSampler3DSpectrum = 0,
        CheckBoard3DSampler = 1,
        MarbleSampler = 2
    };

    union {
        unsigned char unused{};
        ConstantSampler3DSpectrum dataConstantSampler3DSpectrum;
        CheckBoard3DSampler dataCheckBoard3DSampler;
        MarbleSampler dataMarbleSampler;
    };

    TextureSampler3DSpectrumClassType mType;
public:
    TextureSampler3DSpectrumWrapper(): mType(static_cast<TextureSampler3DSpectrumClassType>(0xff)) {};

    explicit TextureSampler3DSpectrumWrapper(const ConstantSampler3DSpectrum& data)
        : dataConstantSampler3DSpectrum(data), mType(TextureSampler3DSpectrumClassType::ConstantSampler3DSpectrum) {}

    explicit TextureSampler3DSpectrumWrapper(const CheckBoard3DSampler& data)
        : dataCheckBoard3DSampler(data), mType(TextureSampler3DSpectrumClassType::CheckBoard3DSampler) {}

    explicit TextureSampler3DSpectrumWrapper(const MarbleSampler& data)
        : dataMarbleSampler(data), mType(TextureSampler3DSpectrumClassType::MarbleSampler) {}

    BOTH TextureSampler3DSpectrumWrapper(const TextureSampler3DSpectrumWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureSampler3DSpectrumWrapper));
    }

    BOTH TextureSampler3DSpectrumWrapper& operator=(const TextureSampler3DSpectrumWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureSampler3DSpectrumWrapper));
        return *this;
    }

    DEVICE Spectrum sample(const TextureMapping3DInfo& info) const {
        switch (mType) {
            case TextureSampler3DSpectrumClassType::ConstantSampler3DSpectrum: return dataConstantSampler3DSpectrum.sample(info);
            case TextureSampler3DSpectrumClassType::CheckBoard3DSampler: return dataCheckBoard3DSampler.sample(info);
            case TextureSampler3DSpectrumClassType::MarbleSampler: return dataMarbleSampler.sample(info);
        }
    }

};
