#pragma once
#include <Texture/TextureMapping.hpp>

class TextureMapping2DWrapper final {
private:
    enum class TextureMapping2DClassType {
        UVMapping = 0,
        SphericalMapping = 1,
        CylindricalMapping = 2,
        PlanarMapping = 3
    };

    union {
        unsigned char unused{};
        UVMapping dataUVMapping;
        SphericalMapping dataSphericalMapping;
        CylindricalMapping dataCylindricalMapping;
        PlanarMapping dataPlanarMapping;
    };

    TextureMapping2DClassType mType;
public:
    TextureMapping2DWrapper(): mType(static_cast<TextureMapping2DClassType>(0xff)) {};

    explicit TextureMapping2DWrapper(const UVMapping& data)
        : dataUVMapping(data), mType(TextureMapping2DClassType::UVMapping) {}

    explicit TextureMapping2DWrapper(const SphericalMapping& data)
        : dataSphericalMapping(data), mType(TextureMapping2DClassType::SphericalMapping) {}

    explicit TextureMapping2DWrapper(const CylindricalMapping& data)
        : dataCylindricalMapping(data), mType(TextureMapping2DClassType::CylindricalMapping) {}

    explicit TextureMapping2DWrapper(const PlanarMapping& data)
        : dataPlanarMapping(data), mType(TextureMapping2DClassType::PlanarMapping) {}

    BOTH TextureMapping2DWrapper(const TextureMapping2DWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureMapping2DWrapper));
    }

    BOTH TextureMapping2DWrapper& operator=(const TextureMapping2DWrapper& rhs) {
        memcpy(this, &rhs, sizeof(TextureMapping2DWrapper));
        return *this;
    }

    DEVICE TextureMapping2DInfo map(const SurfaceInteraction& interaction) const {
        switch (mType) {
            case TextureMapping2DClassType::UVMapping: return dataUVMapping.map(interaction);
            case TextureMapping2DClassType::SphericalMapping: return dataSphericalMapping.map(interaction);
            case TextureMapping2DClassType::CylindricalMapping: return dataCylindricalMapping.map(interaction);
            case TextureMapping2DClassType::PlanarMapping: return dataPlanarMapping.map(interaction);
        }
    }

};
