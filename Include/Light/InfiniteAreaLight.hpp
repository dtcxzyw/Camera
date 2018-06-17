#pragma once
#include <Light/Light.hpp>
#include <Texture/Texture.hpp>
#include <Sampler/Sampling.hpp>
#include <Math/Interaction.hpp>

class InfiniteAreaLightRef final : public LightTag {
private:
    BuiltinSamplerRef<RGBA> mSampler;
    Spectrum mScale;
    Point mCenter;
    float mLength;
    Distribution2DRef mDistribution;
    Transform mToLight;
public:
    InfiniteAreaLightRef(BuiltinSamplerRef<RGBA> sampler, const Spectrum& scale,
        const Distribution2DRef& distribution, const Transform& toLight);
    void preprocess(const Point& center, float radius);
    DEVICE LightingSample sampleLi(const vec2 sample, const Interaction& interaction) const {
        float mapPdf;
        const auto uv = mDistribution.sampleContinuous(sample, mapPdf);
        if (mapPdf == 0.0f) return LightingSample{};

        const auto theta = uv.y * pi<float>(), phi = uv.x * two_pi<float>();
        const auto cosTheta = cos(theta), sinTheta = sin(theta);
        const auto sinPhi = sin(phi), cosPhi = cos(phi);
        const auto wi = inverse(mToLight)(Vector{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta});

        const auto pdf = sinTheta != 0.0f ? mapPdf / (2 * pi<float>() * pi<float>() * sinTheta) : 0.0f;

        return LightingSample{
            wi, Spectrum{mSampler.get(uv), SpectrumType::Illuminant},
            interaction.pos + wi * mLength, pdf
        };
    }

    DEVICE float pdfLi(const Interaction&, const Vector& w) const {
        const auto wi = mToLight(w);
        const auto theta = sphericalTheta(wi), phi = sphericalPhi(wi);
        const auto sinTheta = sin(theta);
        if (sinTheta == 0.0f) return 0.0f;
        return mDistribution.pdf(vec2{phi * one_over_two_pi<float>(), theta * one_over_pi<float>()}) /
            (2.0f * pi<float>() * pi<float>() * sinTheta);
    }

    DEVICE Spectrum le(const Ray& ray) const {
        const auto w = normalize(mToLight(ray.dir));
        const vec2 st(sphericalPhi(w) * one_over_two_pi<float>(),
            sphericalTheta(w) * one_over_pi<float>());
        return Spectrum{mSampler.get(st), SpectrumType::Illuminant};
    }
};

class InfiniteAreaLight final {
private:
    std::shared_ptr<BuiltinMipmapedArray<RGBA>> mArray;
    std::unique_ptr<BuiltinSampler<RGBA>> mSampler;
    Distribution2D mDistribution;
    Spectrum mScale;
    Transform mToLight;
    std::string computeDistribution(const std::string& texture, Stream& stream);
public:
    InfiniteAreaLight(const Transform& trans, const Spectrum& scale, const std::string& texture,
        Stream& stream);
    InfiniteAreaLightRef toRef() const;
};
