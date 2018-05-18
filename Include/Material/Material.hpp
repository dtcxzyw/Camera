#pragma once
#include <BxDF/BxDFWarpper.hpp>
#include <Math/Interaction.hpp>

class Bsdf final {
private:
    Vector mNormal;
    Vector mTangent;
    Vector mBiTangent;
    Interaction mInteraction;
    static constexpr auto maxSize = 4;
    BxDFWarpper mBxDF[maxSize];
    unsigned int mCount;
    float mEta;
    DEVICE Vector toLocal(const Vector& vec) const {
        return {dot(vec, mTangent), dot(vec, mBiTangent), dot(vec, mNormal)};
    }

    DEVICE Vector toWorld(const Vector& vec) const {
        return mat3(mTangent, mBiTangent, mNormal) * vec;
    }

    DEVICE float pdfImpl(const Vector& wo, const Vector& wi, const BxDFType pattern) const {
        if (wo.z == 0.0f) return 0.0f;
        auto pdf = 0.f;
        auto count = 0U;
        for (auto i = 0U; i < mCount; ++i)
            if (mBxDF[i].match(pattern)) {
                ++count;
                pdf += mBxDF[i].pdf(wo, wi);
            }
        return count ? pdf / mCount : 0.0f;
    }

    DEVICE Spectrum fImpl(const Vector& wo, const Vector& worldWo,
        const Vector& wi, const Vector& worldWi, const BxDFType pattern) const {
        if (wo.z == 0.0f) return Spectrum{};
        const auto flags = pattern | (dot(worldWo, mNormal) * dot(worldWi, mNormal) > 0.0f
                                          ? BxDFType::Reflection
                                          : BxDFType::Transmission);
        Spectrum f;
        for (auto i = 0U; i < mCount; ++i)
            if (mBxDF[i].match(flags))
                f += mBxDF[i].f(wo, wi);
        return f;
    }

public:
    DEVICE explicit Bsdf(const Interaction& interaction, const float eta)
        : mNormal(interaction.normal), mTangent(interaction.dpdu),
        mBiTangent(interaction.dpdv), mInteraction(interaction), mCount(0), mEta(eta) {}

    DEVICE float getEta() const {
        return mEta;
    }

    template<typename BxDF>
    DEVICE void add(const BxDF& bxDF) {
        mBxDF[mCount++] = BxDFWarpper{ bxDF };
    }

    DEVICE unsigned int match(const BxDFType type) const {
        unsigned int res = 0;
        for (auto i = 0U; i < mCount; ++i)
            res += mBxDF[i].match(type);
        return res;
    }

    DEVICE float pdf(const Vector& worldWo, const Vector& worldWi, const BxDFType pattern) const {
        return pdfImpl(toLocal(worldWo), toLocal(worldWi), pattern);
    }

    DEVICE Spectrum f(const Vector& worldWo, const Vector& worldWi, const BxDFType pattern) const {
        return fImpl(toLocal(worldWo), worldWo, toLocal(worldWi), worldWi, pattern);
    }

    DEVICE BxDFSample sampleF(const Vector& worldWo, const vec2 sample, const BxDFType pattern) const {
        const auto count = match(pattern);
        if (count == 0) return {};
        const auto nth = min(static_cast<unsigned int>(sample.x * count), count - 1U);

        auto cur = nth, id = 0U;
        for (auto i = 0U; i < mCount; ++i)
            if (mBxDF[i].match(pattern) && cur++ == nth) {
                id = i;
                break;
            }

        const vec2 sampleRemapped(fmin(sample.x * count - nth, 1.0f), sample.y);

        const auto wo = toLocal(worldWo);
        if (wo.z == 0) return {};
        const auto res = mBxDF[id].sampleF(wo, sampleRemapped);
        if (res.pdf == 0.0f)return {};
        const auto wi = toWorld(Vector(res.wi));
        if (mBxDF[id].match(BxDFType::Specular))return {wi, res.f, res.type, res.pdf / count};
        return {
            wi, fImpl(wo, worldWo, Vector(res.wi), wi, pattern), res.type,
            pdfImpl(wo, Vector(res.wi), pattern) / count
        };
    }
    DEVICE const Interaction& getInteraction() const {
        return mInteraction;
    }
};

struct Material {};
