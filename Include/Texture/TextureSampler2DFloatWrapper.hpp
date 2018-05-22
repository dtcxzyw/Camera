#pragma once
#include <Texture/TextureMapping.hpp>
#include <Texture/TextureSampler.hpp>

class TextureSampler2DFloatWrapper final {
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
    TextureSampler2DFloatWrapper(): mType(static_cast<TextureSampler2DFloatClassType>(15)) {};

    explicit TextureSampler2DFloatWrapper(const ConstantSampler2DFloat& data)
        : dataConstantSampler2DFloat(data), mType(TextureSampler2DFloatClassType::ConstantSampler2DFloat) {}

    explicit TextureSampler2DFloatWrapper(const TextureMapFloatSampler& data)
        : dataTextureMapFloatSampler(data), mType(TextureSampler2DFloatClassType::TextureMapFloatSampler) {}

    TextureSampler2DFloatWrapper(const TextureSampler2DFloatWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureSampler2DFloatWrapper));
    }

    TextureSampler2DFloatWrapper& operator=(const TextureSampler2DFloatWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureSampler2DFloatWrapper));
    return *this;
    }

    DEVICE float sample(const TextureMapping2DInfo& info) const {
        switch (mType) {
            case TextureSampler2DFloatClassType::ConstantSampler2DFloat: return dataConstantSampler2DFloat.sample(info);
            case TextureSampler2DFloatClassType::TextureMapFloatSampler: return dataTextureMapFloatSampler.sample(info);
        }
    }

};
