#pragma once
#include <BxDF/BxDFWarpper.hpp>
#include <Math/Interaction.hpp>

class Bsdf final {
private:
    Vector mNormal;
    Vector mTangent;
    Vector mBiTangent;
    static constexpr auto maxSize = 4;
    BxDFWarpper mBxDF[maxSize];
    unsigned int mCount;
    float mEta;
    CUDA Vector toLocal(const Vector& vec) const {
        return {dot(vec, mTangent), dot(vec, mBiTangent), dot(vec, mNormal)};
    }

    CUDA Vector toWorld(const Vector& vec) const {
        return mat3(mTangent, mBiTangent, mNormal) * vec;
    }

    CUDA float pdfImpl(const Vector& wo, const Vector& wi, const BxDFType partten) const {
        if (wo.z == 0.0f) return 0.0f;
        auto pdf = 0.f;
        auto count = 0U;
        for (auto i = 0U; i < mCount; ++i)
            if (mBxDF[i].match(partten)) {
                ++count;
                pdf += mBxDF[i].pdf(wo, wi);
            }
        return count ? pdf / mCount : 0.0f;
    }

    CUDA Spectrum fImpl(const Vector& wo, const Vector& worldWo,
        const Vector& wi, const Vector& worldWi, const BxDFType partten) const {
        if (wo.z == 0.0f) return Spectrum{};
        const auto flags = partten | (dot(worldWo, mNormal) * dot(worldWi, mNormal) > 0.0f
                                          ? BxDFType::Reflection
                                          : BxDFType::Transmission);
        Spectrum f;
        for (auto i = 0U; i < mCount; ++i)
            if (mBxDF[i].match(flags))
                f += mBxDF[i].f(wo, wi);
        return f;
    }

public:
    CUDA explicit Bsdf(const Interaction& interaction, const float eta)
        : mNormal(interaction.normal), mTangent(interaction.dpdu),
        mBiTangent(interaction.dpdv), mCount(0), mEta(eta) {}

    CUDA float getEta() const {
        return mEta;
    }

    CUDA void add(const BxDFWarpper& bxDF) {
        mBxDF[mCount++] = bxDF;
    }

    CUDA unsigned int match(const BxDFType type) const {
        unsigned int res = 0;
        for (auto i = 0U; i < mCount; ++i)
            res += mBxDF[i].match(type);
        return res;
    }

    CUDA float pdf(const Vector& worldWo, const Vector& worldWi, const BxDFType partten) const {
        return pdfImpl(toLocal(worldWo), toLocal(worldWi), partten);
    }

    CUDA Spectrum f(const Vector& worldWo, const Vector& worldWi, const BxDFType partten) const {
        return fImpl(toLocal(worldWo), worldWo, toLocal(worldWi), worldWi, partten);
    }

    BxDFSample sampleF(const Vector& worldWo, const vec2 sample, const BxDFType partten) const {
        const auto count = match(partten);
        if (count == 0) return {};
        const auto nth = min(static_cast<unsigned int>(sample.x * count), count - 1U);

        auto cur = nth, id = 0U;
        for (auto i = 0U; i < mCount; ++i)
            if (mBxDF[i].match(partten) && cur++ == nth) {
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
            wi, fImpl(wo, worldWo, Vector(res.wi), wi, partten), res.type,
            pdfImpl(wo, Vector(res.wi), partten) / count
        };
    }
};

class MaterialRef {
public:
    virtual ~MaterialRef() = default;
    virtual void computeScatteringFunctions(Bsdf& bsdf, 
        TransportMode mode = TransportMode::Radiance) const = 0;
};
