#pragma once
#include <Texture/TextureMapping.hpp>

class TextureMapping2DWarpper final {
private:
    enum class TextureMapping2DClassType {
        UVMapping,
        SphericalMapping,
        CylindricalMapping,
        PlanarMapping
    };

    union {
        char unused{};
        UVMapping dataUVMapping;
        SphericalMapping dataSphericalMapping;
        CylindricalMapping dataCylindricalMapping;
        PlanarMapping dataPlanarMapping;
    };

    TextureMapping2DClassType mType;
public:
    TextureMapping2DWarpper(): mType(static_cast<TextureMapping2DClassType>(15)) {};

    explicit TextureMapping2DWarpper(const UVMapping& data)
        : dataUVMapping(data), mType(TextureMapping2DClassType::UVMapping) {}

    explicit TextureMapping2DWarpper(const SphericalMapping& data)
        : dataSphericalMapping(data), mType(TextureMapping2DClassType::SphericalMapping) {}

    explicit TextureMapping2DWarpper(const CylindricalMapping& data)
        : dataCylindricalMapping(data), mType(TextureMapping2DClassType::CylindricalMapping) {}

    explicit TextureMapping2DWarpper(const PlanarMapping& data)
        : dataPlanarMapping(data), mType(TextureMapping2DClassType::PlanarMapping) {}

    DEVICE TextureMapping2DInfo map(const Interaction& interaction) const {
        switch (mType) {
            case TextureMapping2DClassType::UVMapping: return dataUVMapping.map(interaction);
            case TextureMapping2DClassType::SphericalMapping: return dataSphericalMapping.map(interaction);
            case TextureMapping2DClassType::CylindricalMapping: return dataCylindricalMapping.map(interaction);
            case TextureMapping2DClassType::PlanarMapping: return dataPlanarMapping.map(interaction);
        }
    }

};
