#pragma once
#include <BxDF/BxDF.hpp>
#include <BxDF/Microfact.hpp>

class BxDFWrapper final {
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
        SpecularReflection dataSpecularReflection;
        SpecularTransmission dataSpecularTransmission;
        FresnelSpecular dataFresnelSpecular;
        LambertianReflection dataLambertianReflection;
        OrenNayar dataOrenNayar;
        MicrofacetReflection dataMicrofacetReflection;
        MicrofacetTransmission dataMicrofacetTransmission;
        FresnelBlend dataFresnelBlend;
    };

    BxDFClassType mType;
public:
    DEVICE BxDFWrapper(): mType(static_cast<BxDFClassType>(15)) {};

    DEVICE explicit BxDFWrapper(const SpecularReflection& data)
        : dataSpecularReflection(data), mType(BxDFClassType::SpecularReflection) {}

    DEVICE explicit BxDFWrapper(const SpecularTransmission& data)
        : dataSpecularTransmission(data), mType(BxDFClassType::SpecularTransmission) {}

    DEVICE explicit BxDFWrapper(const FresnelSpecular& data)
        : dataFresnelSpecular(data), mType(BxDFClassType::FresnelSpecular) {}

    DEVICE explicit BxDFWrapper(const LambertianReflection& data)
        : dataLambertianReflection(data), mType(BxDFClassType::LambertianReflection) {}

    DEVICE explicit BxDFWrapper(const OrenNayar& data)
        : dataOrenNayar(data), mType(BxDFClassType::OrenNayar) {}

    DEVICE explicit BxDFWrapper(const MicrofacetReflection& data)
        : dataMicrofacetReflection(data), mType(BxDFClassType::MicrofacetReflection) {}

    DEVICE explicit BxDFWrapper(const MicrofacetTransmission& data)
        : dataMicrofacetTransmission(data), mType(BxDFClassType::MicrofacetTransmission) {}

    DEVICE explicit BxDFWrapper(const FresnelBlend& data)
        : dataFresnelBlend(data), mType(BxDFClassType::FresnelBlend) {}

    DEVICE float pdf(const Vector& wo, const Vector& wi) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return dataSpecularReflection.pdf(wo, wi);
            case BxDFClassType::SpecularTransmission: return dataSpecularTransmission.pdf(wo, wi);
            case BxDFClassType::FresnelSpecular: return dataFresnelSpecular.pdf(wo, wi);
            case BxDFClassType::LambertianReflection: return dataLambertianReflection.pdf(wo, wi);
            case BxDFClassType::OrenNayar: return dataOrenNayar.pdf(wo, wi);
            case BxDFClassType::MicrofacetReflection: return dataMicrofacetReflection.pdf(wo, wi);
            case BxDFClassType::MicrofacetTransmission: return dataMicrofacetTransmission.pdf(wo, wi);
            case BxDFClassType::FresnelBlend: return dataFresnelBlend.pdf(wo, wi);
        }
    }

    DEVICE Spectrum f(const Vector& wo, const Vector& wi) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return dataSpecularReflection.f(wo, wi);
            case BxDFClassType::SpecularTransmission: return dataSpecularTransmission.f(wo, wi);
            case BxDFClassType::FresnelSpecular: return dataFresnelSpecular.f(wo, wi);
            case BxDFClassType::LambertianReflection: return dataLambertianReflection.f(wo, wi);
            case BxDFClassType::OrenNayar: return dataOrenNayar.f(wo, wi);
            case BxDFClassType::MicrofacetReflection: return dataMicrofacetReflection.f(wo, wi);
            case BxDFClassType::MicrofacetTransmission: return dataMicrofacetTransmission.f(wo, wi);
            case BxDFClassType::FresnelBlend: return dataFresnelBlend.f(wo, wi);
        }
    }

    DEVICE BxDFSample sampleF(const Vector& wo, const vec2 sample) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return dataSpecularReflection.sampleF(wo, sample);
            case BxDFClassType::SpecularTransmission: return dataSpecularTransmission.sampleF(wo, sample);
            case BxDFClassType::FresnelSpecular: return dataFresnelSpecular.sampleF(wo, sample);
            case BxDFClassType::LambertianReflection: return dataLambertianReflection.sampleF(wo, sample);
            case BxDFClassType::OrenNayar: return dataOrenNayar.sampleF(wo, sample);
            case BxDFClassType::MicrofacetReflection: return dataMicrofacetReflection.sampleF(wo, sample);
            case BxDFClassType::MicrofacetTransmission: return dataMicrofacetTransmission.sampleF(wo, sample);
            case BxDFClassType::FresnelBlend: return dataFresnelBlend.sampleF(wo, sample);
        }
    }

    DEVICE bool match(const BxDFType pattern) const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return dataSpecularReflection.match(pattern);
            case BxDFClassType::SpecularTransmission: return dataSpecularTransmission.match(pattern);
            case BxDFClassType::FresnelSpecular: return dataFresnelSpecular.match(pattern);
            case BxDFClassType::LambertianReflection: return dataLambertianReflection.match(pattern);
            case BxDFClassType::OrenNayar: return dataOrenNayar.match(pattern);
            case BxDFClassType::MicrofacetReflection: return dataMicrofacetReflection.match(pattern);
            case BxDFClassType::MicrofacetTransmission: return dataMicrofacetTransmission.match(pattern);
            case BxDFClassType::FresnelBlend: return dataFresnelBlend.match(pattern);
        }
    }

};
