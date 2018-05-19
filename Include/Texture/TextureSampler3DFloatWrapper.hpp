#pragma once
#include <Texture/TextureMapping.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler3DFloatWrapper final {
private:
    enum class TextureSampler3DFloatClassType {
        ConstantSampler3DFloat,
        FbmSampler,
        TurbulenceSampler,
        WindyWavesSampler
    };

    union {
        char unused{};
        ConstantSampler3DFloat dataConstantSampler3DFloat;
        FbmSampler dataFbmSampler;
        TurbulenceSampler dataTurbulenceSampler;
        WindyWavesSampler dataWindyWavesSampler;
    };

    TextureSampler3DFloatClassType mType;
public:
    TextureSampler3DFloatWrapper(): mType(static_cast<TextureSampler3DFloatClassType>(15)) {};

    explicit TextureSampler3DFloatWrapper(const ConstantSampler3DFloat& data)
        : dataConstantSampler3DFloat(data), mType(TextureSampler3DFloatClassType::ConstantSampler3DFloat) {}

    explicit TextureSampler3DFloatWrapper(const FbmSampler& data)
        : dataFbmSampler(data), mType(TextureSampler3DFloatClassType::FbmSampler) {}

    explicit TextureSampler3DFloatWrapper(const TurbulenceSampler& data)
        : dataTurbulenceSampler(data), mType(TextureSampler3DFloatClassType::TurbulenceSampler) {}

    explicit TextureSampler3DFloatWrapper(const WindyWavesSampler& data)
        : dataWindyWavesSampler(data), mType(TextureSampler3DFloatClassType::WindyWavesSampler) {}

    DEVICE float sample(const TextureMapping3DInfo& info) const {
        switch (mType) {
            case TextureSampler3DFloatClassType::ConstantSampler3DFloat: return dataConstantSampler3DFloat.sample(info);
            case TextureSampler3DFloatClassType::FbmSampler: return dataFbmSampler.sample(info);
            case TextureSampler3DFloatClassType::TurbulenceSampler: return dataTurbulenceSampler.sample(info);
            case TextureSampler3DFloatClassType::WindyWavesSampler: return dataWindyWavesSampler.sample(info);
        }
    }

};
