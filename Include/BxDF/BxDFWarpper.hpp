#pragma once
#include <BxDF/BxDF.hpp>
#include <BxDF/Microfact.hpp>

class BxDFWarpper final {
private:
    enum class BxDFClassType {
        SpecularReflection,
        SpecularTransmission,
        FresnelSpecular,
        LambertianReflection,
        OrenNayar,
        MicrofacetReflection,
        MicrofacetTransmission,
        FresnelBlend
    };

    union {
        char unused{};
        SpecularReflection bxdfSpecularReflection;
        SpecularTransmission bxdfSpecularTransmission;
        FresnelSpecular bxdfFresnelSpecular;
        LambertianReflection bxdfLambertianReflection;
        OrenNayar bxdfOrenNayar;
        MicrofacetReflection bxdfMicrofacetReflection;
        MicrofacetTransmission bxdfMicrofacetTransmission;
        FresnelBlend bxdfFresnelBlend;
    };

    BxDFClassType mType;
public:
    CUDA BxDFWarpper(): mType(static_cast<BxDFClassType>(15)) {};

    CUDA explicit BxDFWarpper(const SpecularReflection& data)
        : bxdfSpecularReflection(data), mType(BxDFClassType::SpecularReflection) {}

    CUDA explicit BxDFWarpper(const SpecularTransmission& data)
        : bxdfSpecularTransmission(data), mType(BxDFClassType::SpecularTransmission) {}

    CUDA explicit BxDFWarpper(const FresnelSpecular& data)
        : bxdfFresnelSpecular(data), mType(BxDFClassType::FresnelSpecular) {}

    CUDA explicit BxDFWarpper(const LambertianReflection& data)
        : bxdfLambertianReflection(data), mType(BxDFClassType::LambertianReflection) {}

    CUDA explicit BxDFWarpper(const OrenNayar& data)
        : bxdfOrenNayar(data), mType(BxDFClassType::OrenNayar) {}

    CUDA explicit BxDFWarpper(const MicrofacetReflection& data)
        : bxdfMicrofacetReflection(data), mType(BxDFClassType::MicrofacetReflection) {}

    CUDA explicit BxDFWarpper(const MicrofacetTransmission& data)
        : bxdfMicrofacetTransmission(data), mType(BxDFClassType::MicrofacetTransmission) {}

    CUDA explicit BxDFWarpper(const FresnelBlend& data)
        : bxdfFresnelBlend(data), mType(BxDFClassType::FresnelBlend) {}

    CUDA float pdf(const Vector& wo, const Vector& wi) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return bxdfSpecularReflection.pdf(wo, wi);
            case BxDFClassType::SpecularTransmission: return bxdfSpecularTransmission.pdf(wo, wi);
            case BxDFClassType::FresnelSpecular: return bxdfFresnelSpecular.pdf(wo, wi);
            case BxDFClassType::LambertianReflection: return bxdfLambertianReflection.pdf(wo, wi);
            case BxDFClassType::OrenNayar: return bxdfOrenNayar.pdf(wo, wi);
            case BxDFClassType::MicrofacetReflection: return bxdfMicrofacetReflection.pdf(wo, wi);
            case BxDFClassType::MicrofacetTransmission: return bxdfMicrofacetTransmission.pdf(wo, wi);
            case BxDFClassType::FresnelBlend: return bxdfFresnelBlend.pdf(wo, wi);
        }
    }

    CUDA Spectrum f(const Vector& wo, const Vector& wi) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return bxdfSpecularReflection.f(wo, wi);
            case BxDFClassType::SpecularTransmission: return bxdfSpecularTransmission.f(wo, wi);
            case BxDFClassType::FresnelSpecular: return bxdfFresnelSpecular.f(wo, wi);
            case BxDFClassType::LambertianReflection: return bxdfLambertianReflection.f(wo, wi);
            case BxDFClassType::OrenNayar: return bxdfOrenNayar.f(wo, wi);
            case BxDFClassType::MicrofacetReflection: return bxdfMicrofacetReflection.f(wo, wi);
            case BxDFClassType::MicrofacetTransmission: return bxdfMicrofacetTransmission.f(wo, wi);
            case BxDFClassType::FresnelBlend: return bxdfFresnelBlend.f(wo, wi);
        }
    }

    CUDA BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return bxdfSpecularReflection.sampleF(wo, sample);
            case BxDFClassType::SpecularTransmission: return bxdfSpecularTransmission.sampleF(wo, sample);
            case BxDFClassType::FresnelSpecular: return bxdfFresnelSpecular.sampleF(wo, sample);
            case BxDFClassType::LambertianReflection: return bxdfLambertianReflection.sampleF(wo, sample);
            case BxDFClassType::OrenNayar: return bxdfOrenNayar.sampleF(wo, sample);
            case BxDFClassType::MicrofacetReflection: return bxdfMicrofacetReflection.sampleF(wo, sample);
            case BxDFClassType::MicrofacetTransmission: return bxdfMicrofacetTransmission.sampleF(wo, sample);
            case BxDFClassType::FresnelBlend: return bxdfFresnelBlend.sampleF(wo, sample);
        }
    }

    CUDA bool match(const BxDFType partten) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return bxdfSpecularReflection.match(partten);
            case BxDFClassType::SpecularTransmission: return bxdfSpecularTransmission.match(partten);
            case BxDFClassType::FresnelSpecular: return bxdfFresnelSpecular.match(partten);
            case BxDFClassType::LambertianReflection: return bxdfLambertianReflection.match(partten);
            case BxDFClassType::OrenNayar: return bxdfOrenNayar.match(partten);
            case BxDFClassType::MicrofacetReflection: return bxdfMicrofacetReflection.match(partten);
            case BxDFClassType::MicrofacetTransmission: return bxdfMicrofacetTransmission.match(partten);
            case BxDFClassType::FresnelBlend: return bxdfFresnelBlend.match(partten);
        }
    }

};
