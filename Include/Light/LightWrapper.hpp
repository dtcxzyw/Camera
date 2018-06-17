#pragma once
#include <Light/DeltaPositionLight.hpp>
#include <Light/DiffuseAreaLight.hpp>
#include <Light/DistantLight.hpp>
#include <Light/InfiniteAreaLight.hpp>

class LightWrapper final {
private:
    enum class LightClassType {
        PointLight = 0,
        SpotLight = 1,
        DiffuseAreaLight = 2,
        DistantLight = 3,
        InfiniteAreaLightRef = 4
    };

    union {
        unsigned char unused{};
        PointLight dataPointLight;
        SpotLight dataSpotLight;
        DiffuseAreaLight dataDiffuseAreaLight;
        DistantLight dataDistantLight;
        InfiniteAreaLightRef dataInfiniteAreaLightRef;
    };

    LightClassType mType;
public:
    LightWrapper(): mType(static_cast<LightClassType>(0xff)) {};

    explicit LightWrapper(const PointLight& data)
        : dataPointLight(data), mType(LightClassType::PointLight) {}

    explicit LightWrapper(const SpotLight& data)
        : dataSpotLight(data), mType(LightClassType::SpotLight) {}

    explicit LightWrapper(const DiffuseAreaLight& data)
        : dataDiffuseAreaLight(data), mType(LightClassType::DiffuseAreaLight) {}

    explicit LightWrapper(const DistantLight& data)
        : dataDistantLight(data), mType(LightClassType::DistantLight) {}

    explicit LightWrapper(const InfiniteAreaLightRef& data)
        : dataInfiniteAreaLightRef(data), mType(LightClassType::InfiniteAreaLightRef) {}

    BOTH LightWrapper(const LightWrapper& rhs) {
        memcpy(this, &rhs, sizeof(LightWrapper));
    }

    BOTH LightWrapper& operator=(const LightWrapper& rhs) {
        memcpy(this, &rhs, sizeof(LightWrapper));
        return *this;
    }

    DEVICE Spectrum le(const Ray& ray) const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.le(ray);
            case LightClassType::SpotLight: return dataSpotLight.le(ray);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.le(ray);
            case LightClassType::DistantLight: return dataDistantLight.le(ray);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.le(ray);
        }
    }

    DEVICE bool isDelta() const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.isDelta();
            case LightClassType::SpotLight: return dataSpotLight.isDelta();
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.isDelta();
            case LightClassType::DistantLight: return dataDistantLight.isDelta();
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.isDelta();
        }
    }

    void preprocess(const Point& center, const float radius) {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.preprocess(center, radius);
            case LightClassType::SpotLight: return dataSpotLight.preprocess(center, radius);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.preprocess(center, radius);
            case LightClassType::DistantLight: return dataDistantLight.preprocess(center, radius);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.preprocess(center, radius);
        }
    }

    DEVICE LightingSample sampleLi(const vec2 sample, const Interaction& isect) const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.sampleLi(sample, isect);
            case LightClassType::SpotLight: return dataSpotLight.sampleLi(sample, isect);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.sampleLi(sample, isect);
            case LightClassType::DistantLight: return dataDistantLight.sampleLi(sample, isect);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.sampleLi(sample, isect);
        }
    }

    DEVICE float pdfLi(const Interaction& isect, const Vector& wi) const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.pdfLi(isect, wi);
            case LightClassType::SpotLight: return dataSpotLight.pdfLi(isect, wi);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.pdfLi(isect, wi);
            case LightClassType::DistantLight: return dataDistantLight.pdfLi(isect, wi);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.pdfLi(isect, wi);
        }
    }

    DEVICE Spectrum emitL(const Interaction& isect, const Vector& wi) const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.emitL(isect, wi);
            case LightClassType::SpotLight: return dataSpotLight.emitL(isect, wi);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.emitL(isect, wi);
            case LightClassType::DistantLight: return dataDistantLight.emitL(isect, wi);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.emitL(isect, wi);
        }
    }

    DEVICE bool intersect(const Ray& ray) const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.intersect(ray);
            case LightClassType::SpotLight: return dataSpotLight.intersect(ray);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.intersect(ray);
            case LightClassType::DistantLight: return dataDistantLight.intersect(ray);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.intersect(ray);
        }
    }

    DEVICE bool intersect(const Ray& ray, float& tHit, SurfaceInteraction& isect) const {
        switch (mType) {
            case LightClassType::PointLight: return dataPointLight.intersect(ray, tHit, isect);
            case LightClassType::SpotLight: return dataSpotLight.intersect(ray, tHit, isect);
            case LightClassType::DiffuseAreaLight: return dataDiffuseAreaLight.intersect(ray, tHit, isect);
            case LightClassType::DistantLight: return dataDistantLight.intersect(ray, tHit, isect);
            case LightClassType::InfiniteAreaLightRef: return dataInfiniteAreaLightRef.intersect(ray, tHit, isect);
        }
    }

};
