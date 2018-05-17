#pragma once
#include <BxDF/BxDF.hpp>

template <typename T>
class MicrofactDistributionHelper {
private:
    DEVICE const T& self() const {
        return *static_cast<const T*>(this);
    }

public:
    DEVICE float calcG1(const Vector& w) const {
        return 1.0f / (1.0f + self().calcLambda(w));
    }

    DEVICE float calcG(const Vector& wo, const Vector& wi) const {
        return 1.0f / (1.0f + self().calcLambda(wo) + self().calcLambda(wi));
    }

    DEVICE float pdf(const Vector& wo, const Vector& wh) const {
        return self().calcD(wh) * calcG1(wo) * fabs(dot(wo, wh)) / absCosTheta(wo);
    }
};

class TrowbridgeReitzDistribution final : public MicrofactDistributionHelper<TrowbridgeReitzDistribution> {
private:
    float mAlphaX, mAlphaY;
    DEVICE static float toAlpha(const float roughness) {
        const auto x = std::log(fmax(roughness, 1e-3f));
        return (((0.000640711f * x + 0.0171201f) * x + 0.1734f) * x + 0.819955f) * x + 1.62142f;
    }

    DEVICE static vec2 sample11(const float cosTheta, const vec2 sample) {
        // special case (normal incidence)
        if (cosTheta > 0.9999f) {
            const auto r = sqrt(sample.x / (1.0f - sample.x));
            const auto phi = 6.28318530718 * sample.y;
            return {r * cos(phi), r * sin(phi)};
        }

        const auto sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        const auto tanTheta = sinTheta / cosTheta;
        const auto g1 = 2.0f / (1.0f + sqrt(1.0f + tanTheta * tanTheta));

        // sample slope.x
        const auto a = 2.0f * sample.x / g1 - 1.0f;
        const auto tmp = fmin(1e10f, 1.0f / (a * a - 1.0f));
        const auto b = tanTheta;
        const auto d = sqrt(fmax(b * b * tmp * tmp - (a * a - b * b) * tmp, 0.0f));
        const auto slopeX1 = b * tmp - d;
        const auto slopeX2 = b * tmp + d;
        const auto slopeX = (a < 0.0f | slopeX2 > 1.0f / tanTheta) ? slopeX1 : slopeX2;

        // sample slope.y
        const auto s = sample.y > 0.5f ? 1.0f : -1.0f;
        const auto u2 = 2.0f * fabs(sample.y - 0.5f);
        const auto z =
            (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) /
            (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        return {slopeX, s * z * sqrt(1.0f + slopeX * slopeX)};
    }

public:
    DEVICE TrowbridgeReitzDistribution(const float rx, const float ry)
        : mAlphaX(toAlpha(rx)), mAlphaY(toAlpha(ry)) {}

    DEVICE float calcD(const Vector& wh) const {
        const auto tan2ThetaH = tan2Theta(wh);
        if (std::isinf(tan2ThetaH)) return 0.0f;
        const auto cos4Theta = cos2Theta(wh) * cos2Theta(wh);
        const auto e = (cos2Phi(wh) / (mAlphaX * mAlphaX) + sin2Phi(wh) / (mAlphaY * mAlphaY)) *
            tan2ThetaH;
        return one_over_pi<float>() / (mAlphaX * mAlphaY * cos4Theta * (1 + e) * (1 + e));
    }

    DEVICE float calcLambda(const Vector& w) const {
        const auto alpha2 = cosPhi(w) * mAlphaX * mAlphaX + sinPhi(w) * mAlphaY * mAlphaY;
        const auto tan2ThetaH = tan2Theta(w);
        return 0.5f * (-1.0f + sqrt(1.0f + alpha2 * tan2ThetaH));
    }

    DEVICE Vector sampleWh(const Vector& wo, const vec2 sample) const {
        const auto filp = wo.z < 0.0f;
        const auto wi = filp ? -wo : wo;
        // 1. stretch wi
        const auto wiStretched = glm::normalize(Vector(mAlphaX * wi.x, mAlphaY * wi.y, wi.z));

        // 2. simulate P22_{wi}(slope.x, slope.y, 1, 1)
        auto slope = sample11(cosTheta(wiStretched), sample);

        // 3. rotate
        const auto tmp = cosPhi(wiStretched) * slope.x - sinPhi(wiStretched) * slope.y;
        slope.y = sinPhi(wiStretched) * slope.x + cosPhi(wiStretched) * slope.y;
        slope.x = tmp;

        // 4. unstretch
        slope.x *= mAlphaX;
        slope.y *= mAlphaY;

        // 5. compute normal
        const auto wh = glm::normalize(Vector(-slope.x, -slope.y, 1.0f));
        return filp ? -wh : wh;
    }
};

class MicrofactDistributionWarpper final {
private:
    union {
        TrowbridgeReitzDistribution tr;
    };

public:
    DEVICE explicit MicrofactDistributionWarpper(const TrowbridgeReitzDistribution& tr) : tr(tr) {}
    DEVICE float calcG(const Vector& wo, const Vector& wi) const {
        return tr.calcG(wo, wi);
    }

    DEVICE float pdf(const Vector& wo, const Vector& wh) const {
        return tr.pdf(wo, wh);
    }

    DEVICE float calcD(const Vector& wh) const {
        return tr.calcD(wh);
    }

    DEVICE Vector sampleWh(const Vector& wo, const vec2 sample) const {
        return tr.sampleWh(wo, sample);
    }
};

class MicrofacetReflection final : public BxDFHelper<MicrofacetReflection> {
private:
    Spectrum mReflection;
    FresnelWarpper mFresnel;
    MicrofactDistributionWarpper mDistribution;
public:
    static constexpr auto type = BxDFType::Reflection | BxDFType::Glossy;
    DEVICE MicrofacetReflection(const Spectrum& reflection, const FresnelWarpper& fresnel,
        const MicrofactDistributionWarpper& distribution)
        : mReflection(reflection), mFresnel(fresnel), mDistribution(distribution) {}

    DEVICE float pdf(const Vector& wo, const Vector& wi) const {
        if (wo.z * wi.z < 0.0f)return 0.0f;
        const auto wh = halfVector(wi, wo);
        return mDistribution.pdf(wo, wh) / (4.0f * dot(wo, wh));
    }

    DEVICE Spectrum f(const Vector& wo, const Vector& wi) const {
        const auto cosThetaO = absCosTheta(wo), cosThetaI = absCosTheta(wi);
        if (cosThetaO == 0.0f | cosThetaI == 0.0f | wi == -wo)return Spectrum{};
        const auto wh = halfVector(wi, wo);
        const auto fac = mDistribution.calcD(wh) * mDistribution.calcG(wo, wi)
            / (4.0f * cosThetaO * cosThetaI);
        return mReflection * mFresnel.f(dot(wi, wh)) * fac;
    }

    DEVICE BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        const auto wh = mDistribution.sampleWh(wo, sample);
        const auto wi = glm::reflect(wo, wh);
        if (wi.z * wo.z <= 0.0f)return BxDFSample{};
        return {wi, f(wo, wi), type, mDistribution.pdf(wo, wh) / (4.0f * dot(wo, wh))};
    }
};

class MicrofacetTransmission final : public BxDFHelper<MicrofacetTransmission> {
private:
    Spectrum mTransmission;
    FresnelDielectric mFresnel;
    MicrofactDistributionWarpper mDistribution;
    float mEtaA, mEtaB;
    TransportMode mMode;
public:
    static constexpr auto type = BxDFType::Transmission | BxDFType::Glossy;
    DEVICE MicrofacetTransmission(const Spectrum& transmission, const float etaA, const float etaB,
        const MicrofactDistributionWarpper& distribution, const TransportMode mode)
        : mTransmission(transmission), mFresnel(etaA, etaB), mDistribution(distribution),
        mEtaA(etaA), mEtaB(etaB), mMode(mode) {}

    DEVICE float pdf(const Vector& wo, const Vector& wi) const {
        if (wo.z * wi.z > 0.0f)return 0.0f;
        const auto eta = cosTheta(wo) > 0.0f ? (mEtaB / mEtaA) : (mEtaA / mEtaB);
        const auto wh = glm::normalize(wo + wi * eta);

        const auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        const auto dwhdwi = fabs(eta * eta * dot(wi, wh) / (sqrtDenom * sqrtDenom));
        return mDistribution.pdf(wo, wh) * dwhdwi;
    }

    DEVICE Spectrum f(const Vector& wo, const Vector& wi) const {
        const auto cosThetaO = cosTheta(wo), cosThetaI = cosTheta(wi);
        if (cosThetaO * cosThetaI >= 0.0f) return Spectrum{};

        const auto eta = cosThetaO > 0.0f ? (mEtaB / mEtaA) : (mEtaA / mEtaB);
        auto wh = glm::normalize(wo + wi * eta);
        if (wh.z < 0.0f) wh = -wh;

        const auto fresnel = mFresnel.f(dot(wo, wh));

        const auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        const auto fac = mMode == TransportMode::Radiance ? (1.0f / eta) : 1.0f;
        const auto k = eta * fac / sqrtDenom;

        return (1.0f - fresnel) * mTransmission *
            fabs(mDistribution.calcD(wh) * mDistribution.calcG(wo, wi) * dot(wi, wh) * dot(wo, wh) * k * k
                / (cosThetaI * cosThetaO));
    }

    DEVICE BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        const auto wh = mDistribution.sampleWh(wo, sample);
        const auto eta = cosTheta(wo) > 0.0f ? (mEtaB / mEtaA) : (mEtaA / mEtaB);
        Vector wi;
        if (!refract(wo, wh, eta, wi)) return BxDFSample{};
        return {wi, f(wo, wi), type, pdf(wo, wi)};
    }
};

class FresnelBlend final : public BxDFHelper<FresnelBlend> {
private:
    Spectrum mRd, mRs;
    MicrofactDistributionWarpper mDistribution;

    static DEVICE float pow5(const float d) {
        const auto d2 = d * d;
        return d * d2 * d2;
    }

    DEVICE Spectrum fresnelSchlick(const float cosTheta) const {
        return mRs + pow5(1.0f - cosTheta) * (1.0f - mRs);
    }

public:
    static constexpr auto type = BxDFType::Reflection | BxDFType::Glossy;
    DEVICE FresnelBlend(const Spectrum& rd, const Spectrum& rs,
        const MicrofactDistributionWarpper& distribution)
        : mRd(rd), mRs(rs), mDistribution(distribution) {}

    DEVICE float pdf(const Vector& wo, const Vector& wi) const {
        if (wo.z * wi.z < 0.0f)return 0.0f;
        const auto wh = halfVector(wi, wo);
        return 0.5f * (absCosTheta(wi) * one_over_pi<float>() +
            mDistribution.pdf(wo, wh) / (4.0f * dot(wo, wh)));
    }

    DEVICE Spectrum f(const Vector& wo, const Vector& wi) const {
        if (wo == -wi)return Spectrum{};
        const auto diffuse = (28.f / (23.f * pi<float>())) * mRd * (Spectrum(1.f) - mRs) *
            (1.0f - pow5(1.0f - 0.5f * absCosTheta(wi))) *
            (1.0f - pow5(1.0f - 0.5f * absCosTheta(wo)));
        const auto wh = glm::normalize(wi + wo);
        const auto specular =
            mDistribution.calcD(wh) /
            (4 * fabs(dot(wi, wh)) * fmax(absCosTheta(wi), absCosTheta(wo))) *
            fresnelSchlick(dot(wi, wh));
        return diffuse + specular;
    }

    DEVICE BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        auto u = sample;
        Vector wi;
        if (u[0] < 0.5f) {
            u.x *= 2.0f;
            wi = cosineSampleHemisphere(u);
            if (wo.z < 0.0f) wi.z = -wi.z;
        }
        else {
            u.x = 2.0f * (u.x - 0.5f);
            const auto wh = mDistribution.sampleWh(wo, u);
            wi = reflect(wo, wh);
            if (wo.z * wi.z < 0.0f) return {};
        }
        return {wi, f(wo, wi), type, pdf(wo, wi)};
    }
};
