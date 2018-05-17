#pragma once
#include <Texture/TextureMapping.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler2DFloatWarpper final {
private:
    enum class TextureSampler2DFloatClassType {
        ConstantSampler2DFloat,
        TextureMapFloatSampler
    };

    union {
        char unused{};
        ConstantSampler2DFloat dataConstantSampler2DFloat;
        TextureMapFloatSampler dataTextureMapFloatSampler;
    };

    TextureSampler2DFloatClassType mType;
public:
    TextureSampler2DFloatWarpper(): mType(static_cast<TextureSampler2DFloatClassType>(15)) {};

    explicit TextureSampler2DFloatWarpper(const ConstantSampler2DFloat& data)
        : dataConstantSampler2DFloat(data), mType(TextureSampler2DFloatClassType::ConstantSampler2DFloat) {}

    explicit TextureSampler2DFloatWarpper(const TextureMapFloatSampler& data)
        : dataTextureMapFloatSampler(data), mType(TextureSampler2DFloatClassType::TextureMapFloatSampler) {}

    DEVICE float sample(const TextureMapping2DInfo& info) const {
        switch (mType) {
            case TextureSampler2DFloatClassType::ConstantSampler2DFloat: return dataConstantSampler2DFloat.sample(info);
            case TextureSampler2DFloatClassType::TextureMapFloatSampler: return dataTextureMapFloatSampler.sample(info);
        }
    }

};
