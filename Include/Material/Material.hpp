#pragma once
#include <BxDF/BxDFWrapper.hpp>
#include <Math/Interaction.hpp>

class Bsdf final {
private:
    const SurfaceInteraction& mInteraction;
    const Vector mNormal, mTangent, mBiTangent, mLocalNormal;
    static constexpr auto maxSize = 4;
    BxDFWrapper mBxDF[maxSize];
    uint32_t mCount;
    float mEta;
    DEVICE Vector toLocal(const Vector& vec) const;

    DEVICE Vector toWorld(const Vector& vec) const;

    DEVICE float pdfImpl(const Vector& wo, const Vector& wi, BxDFType pattern) const;

    DEVICE Spectrum fImpl(const Vector& wo, const Vector& worldWo,
        const Vector& wi, const Vector& worldWi, BxDFType pattern) const;

public:
    DEVICE explicit Bsdf(const SurfaceInteraction& interaction);

    DEVICE float getEta() const {
        return mEta;
    }

    DEVICE void setEta(const float eta) {
        mEta = eta;
    }

    template <typename BxDF>
    DEVICE void add(const BxDF& bxDF) {
        mBxDF[mCount++] = BxDFWrapper{bxDF};
    }

    DEVICE uint32_t match(BxDFType type) const;

    DEVICE float pdf(const Vector& worldWo, const Vector& worldWi, BxDFType pattern
        = BxDFType::All) const;

    DEVICE Spectrum f(const Vector& worldWo, const Vector& worldWi, BxDFType pattern
        = BxDFType::All) const;

    DEVICE BxDFSample sampleF(const Vector& worldWo, vec2 sample,
        BxDFType pattern = BxDFType::All) const;

    DEVICE const SurfaceInteraction& getInteraction() const {
        return mInteraction;
    }
};

struct Material {};
