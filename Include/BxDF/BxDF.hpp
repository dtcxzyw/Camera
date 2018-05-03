#pragma once
#include <Math/Geometry.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <BxDF/Fresnel.hpp>
#include <Sampler/Samping.hpp>

enum class BxDFType {
    Reflection = 1,
    Transmission = 2,
    Diffuse = 4,
    Glossy = 8,
    Specular = 16,
    All = 31
};

enum class TransportMode {
    Radiance,
    Importance
};

BOTH BxDFType operator&(const BxDFType a, const BxDFType b) {
    return static_cast<BxDFType>(static_cast<unsigned int>(a) & static_cast<unsigned int>(b));
}

constexpr BOTH BxDFType operator|(const BxDFType a, const BxDFType b) {
    return static_cast<BxDFType>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

template <BxDFType Type>
BOTH bool matchPattern(const BxDFType pattern) {
    return (Type & pattern) == Type;
}

struct BxDFSample final {
    Normal wi;
    float pdf;
    Spectrum f;
    BxDFType type;
    CUDA BxDFSample() : pdf(0.0f), type(static_cast<BxDFType>(0)) {}
    CUDA BxDFSample(const Vector& wi, const Spectrum& f, const BxDFType type, const float pdf = 1.0f)
        : wi(makeNormalUnsafe(wi)), pdf(pdf), f(f), type(type) {}
};

CUDAINLINE float cosTheta(const Vector& v) {
    return v.z;
}

CUDAINLINE float absCosTheta(const Vector& v) {
    return fabs(v.z);
}

CUDAINLINE float cos2Theta(const Vector& v) {
    return v.z * v.z;
}

CUDAINLINE float sin2Theta(const Vector& v) {
    return 1.0f - cos2Theta(v);
}

CUDAINLINE float sinTheta(const Vector& v) {
    return sqrt(sin2Theta(v));
}

CUDAINLINE float tan2Theta(const Vector& v) {
    return sin2Theta(v) / cos2Theta(v);
}

CUDAINLINE float tanTheta(const Vector& v) {
    return sinTheta(v) / cosTheta(v);
}

CUDAINLINE float cosPhi(const Vector& v) {
    const auto sinThetaV = sinTheta(v);
    return sinThetaV == 0.0f ? 1.0f : v.x / sinThetaV;
}

CUDAINLINE float cos2Phi(const Vector& v) {
    return cosPhi(v)*cosPhi(v);
}

CUDAINLINE float sinPhi(const Vector& v) {
    const auto sinThetaV = sinTheta(v);
    return sinThetaV == 0.0f ? 0.0f : v.y / sinThetaV;
}

CUDAINLINE float sin2Phi(const Vector& v) {
    return sinPhi(v)*sinPhi(v);
}

template <typename T>
struct BxDFHelper {

    CUDA bool match(const BxDFType partten) const {
        return matchPattern<T::type>(partten);
    }

    CUDA Spectrum f(const Vector&, const Vector&) const {
        return Spectrum{};
    }

    CUDA float pdf(const Vector& wo, const Vector& wi) const {
        return wo.z * wi.z > 0.0f ? absCosTheta(wi) * one_over_pi<float>() : 0.0f;
    }

    CUDA BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        auto&& self = *static_cast<const T*>(this);
        auto wi = cosineSampleHemisphere(sample);
        if (wo.z < 0.0f)wi.z = -wi.z;
        return {wi, self.f(wo, wi), self.pdf(wo, wi)};
    }
};

class SpecularReflection final : public BxDFHelper<SpecularReflection> {
private:
    Spectrum mReflection;
    FresnelWarpper mFresnel;
public:
    static constexpr auto type = BxDFType::Reflection | BxDFType::Specular;
    CUDA SpecularReflection(const Spectrum& reflection, const FresnelWarpper& fresnel)
        : mReflection(reflection), mFresnel(fresnel) {}

    CUDA float pdf(const Vector&, const Vector&) const {
        return 0.0f;
    }

    CUDA BxDFSample sampleF(const Vector& wo, const vec2) const {
        const Vector wi = {-wo.x, -wo.y, wo.z};
        return {wi, mFresnel.f(cosTheta(wi)) * mReflection / absCosTheta(wi), type};
    }
};

class SpecularTransmission final : public BxDFHelper<SpecularTransmission> {
private:
    Spectrum mTransmission;
    FresnelDielectric mFresnel;
    TransportMode mMode;
    float mEtaA, mEtaB;
public:
    static constexpr auto type = BxDFType::Transmission | BxDFType::Specular;
    CUDA SpecularTransmission(const Spectrum& transmission, const float etaA, const float etaB,
        const TransportMode mode) : mTransmission(transmission), mFresnel(etaA, etaB),
        mMode(mode), mEtaA(etaA), mEtaB(etaB) {}

    CUDA float pdf(const Vector&, const Vector&) const {
        return 0.0f;
    }

    CUDA BxDFSample sampleF(const Vector& wo, const vec2) const {
        const auto entering = wo.z > 0.0f;
        const auto etaI = entering ? mEtaA : mEtaB;
        const auto eatT = entering ? mEtaB : mEtaA;
        const auto eta = etaI / eatT;
        Vector wi;
        if (!refract(wo, faceForward(Vector{0.0f, 0.0f, 1.0f}, wo), eta, wi))return {};
        auto ft = mTransmission * (1.0f - mFresnel.f(cosTheta(wi)));
        if (mMode == TransportMode::Radiance)ft *= eta * eta;
        return {wi, ft, type};
    }
};

class FresnelSpecular final : public BxDFHelper<FresnelSpecular> {
private:
    Spectrum mReflection, mTransmission;
    FresnelDielectric mFresnel;
    TransportMode mMode;
    float mEtaA, mEtaB;
public:
    static constexpr auto type = BxDFType::Reflection | BxDFType::Transmission | BxDFType::Specular;
    CUDA FresnelSpecular(const Spectrum& reflection, const Spectrum& transmission,
        const float etaA, const float etaB, const TransportMode mode)
        : mReflection(reflection), mTransmission(transmission), mFresnel(etaA, etaB),
        mMode(mode), mEtaA(etaA), mEtaB(etaB) {}

    CUDA float pdf(const Vector&, const Vector&) const {
        return 0.0f;
    }

    CUDA BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        const auto f = mFresnel.f(cosTheta(wo));
        if (sample.x < f) {
            const Vector wi = {-wo.x, -wo.y, wo.z};
            return {
                wi, f * mReflection / absCosTheta(wi),
                BxDFType::Reflection | BxDFType::Specular, f
            };
        }
        else {
            const auto entering = cosTheta(wo) > 0.0f;
            const auto etaI = entering ? mEtaA : mEtaB;
            const auto eatT = entering ? mEtaB : mEtaA;
            const auto eta = etaI / eatT;
            Vector wi;
            if (!refract(wo, faceForward(Vector{0.0f, 0.0f, 1.0f}, wo), eta, wi))return {};
            const auto pdf = 1.0f - f;
            auto ft = mTransmission * pdf;
            if (mMode == TransportMode::Radiance)ft *= eta * eta;
            return {wi, ft / absCosTheta(wi), BxDFType::Transmission | BxDFType::Specular, pdf};
        }
    }
};

class LambertianReflection final : public BxDFHelper<LambertianReflection> {
private:
    Spectrum mReflection;
public:
    static constexpr auto type = BxDFType::Reflection | BxDFType::Diffuse;
    CUDA explicit LambertianReflection(const Spectrum& reflection): mReflection(reflection) {}
    CUDA Spectrum f(const Vector&, const Vector&) const {
        return mReflection * one_over_pi<float>();
    }
};

class OrenNayar final : public BxDFHelper<OrenNayar> {
private:
    Spectrum mReflection;
    float mA, mB;
public:
    static constexpr auto type = BxDFType::Reflection | BxDFType::Diffuse;
    CUDA OrenNayar(const Spectrum& reflection, float sigma)
        : mReflection(reflection) {
        sigma = glm::radians(sigma);
        const auto sigma2 = sigma * sigma;
        mA = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        mB = 0.45f + sigma2 / (sigma2 + 0.09f);
    }

    CUDA Spectrum f(const Vector& wo, const Vector& wi) const {
        const auto sinThetaO = sinTheta(wo);
        const auto sinThetaI = sinTheta(wi);
        auto maxCos = 0.0f;
        if (sinThetaI > 1e-4f & sinThetaO > 1e-4f) {
            const auto sinPhiI = sinPhi(wi), sinPhiO = sinPhi(wo);
            const auto cosPhiI = cosPhi(wi), cosPhiO = cosPhi(wo);
            maxCos = fmax(0.0f, cosPhiI * cosPhiO + sinPhiI * sinPhiO);
        }

        float sinAlpha, tanBeta;
        if (absCosTheta(wi) > absCosTheta(wo)) {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / absCosTheta(wi);
        }
        else {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / absCosTheta(wo);
        }

        return mReflection * (one_over_pi<float>() * (mA + mB * maxCos * sinAlpha * tanBeta));
    }
};
