#pragma once
#include <BxDF/BxDF.hpp>
#include <BxDF/Microfact.hpp>

class BxDFWrapper final {
private:
    enum class BxDFClassType {
        SpecularReflection = 0,
        SpecularTransmission = 1,
        FresnelSpecular = 2,
        LambertianReflection = 3,
        OrenNayar = 4,
        MicrofacetReflection = 5,
        MicrofacetTransmission = 6,
        FresnelBlend = 7
    };

    union {
        unsigned char unused{};
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
    DEVICE BxDFWrapper(): mType(static_cast<BxDFClassType>(0xff)) {};

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

    BOTH BxDFWrapper(const BxDFWrapper& rhs) {
        memcpy(this, &rhs, sizeof(BxDFWrapper));
    }

    BOTH BxDFWrapper& operator=(const BxDFWrapper& rhs) {
        memcpy(this, &rhs, sizeof(BxDFWrapper));
        return *this;
    }

    DEVICE BxDFType getType() const {
        switch (mType) {
            case BxDFClassType::SpecularReflection: return dataSpecularReflection.getType();
            case BxDFClassType::SpecularTransmission: return dataSpecularTransmission.getType();
            case BxDFClassType::FresnelSpecular: return dataFresnelSpecular.getType();
            case BxDFClassType::LambertianReflection: return dataLambertianReflection.getType();
            case BxDFClassType::OrenNayar: return dataOrenNayar.getType();
            case BxDFClassType::MicrofacetReflection: return dataMicrofacetReflection.getType();
            case BxDFClassType::MicrofacetTransmission: return dataMicrofacetTransmission.getType();
            case BxDFClassType::FresnelBlend: return dataFresnelBlend.getType();
        }
    }

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
