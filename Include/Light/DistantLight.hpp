#pragma once
#include <Light/Light.hpp>
#include <Math/Interaction.hpp>

class DistantLight final : public LightTag {
private:
    Vector mDirection, mOffset;
    Spectrum mIllumination;
public:
    DistantLight() = default;

    DistantLight(const Transform& trans, const Vector& dir, const Spectrum& illumination)
        : mDirection(normalize(trans(dir))), mOffset(mDirection), mIllumination(illumination) {}

    void preprocess(const Point&, const float radius) {
        mOffset *= 2.0f * radius;
    }

    DEVICE LightingSample sampleLi(const vec2, const Interaction& interaction) const {
        return LightingSample{mDirection, mIllumination, interaction.pos + mOffset};
    }

    DEVICE bool isDelta() const {
        return true;
    }
};
