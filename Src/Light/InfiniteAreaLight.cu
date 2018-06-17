#include <Light/InfiniteAreaLight.hpp>
#include <IO/Image.hpp>

InfiniteAreaLightRef::InfiniteAreaLightRef(const BuiltinSamplerRef<RGBA> sampler,
    const Spectrum& scale, const Distribution2DRef& distribution, const Transform& toLight)
    : mSampler(sampler), mScale(scale), mLength(0), mDistribution(distribution), mToLight(toLight) {}

void InfiniteAreaLightRef::preprocess(const Point& center, const float radius) {
    mCenter = center;
    mLength = 2.0f * radius;
}

std::string InfiniteAreaLight::computeDistribution(const std::string& texture, Stream& stream) {
    //TODO:compute Distribution

    return texture + ".distrib";
}

InfiniteAreaLight::InfiniteAreaLight(const Transform& trans, const Spectrum& scale,
    const std::string& texture, Stream& stream)
    : mArray(loadMipmapedRGBA(texture, stream)),
    mSampler(std::make_unique<BuiltinSampler<RGBA>>(*mArray)),
    mDistribution(computeDistribution(texture, stream)),
    mScale(scale), mToLight(inverse(trans)) {}

InfiniteAreaLightRef InfiniteAreaLight::toRef() const {
    return {mSampler->toRef(), mScale, mDistribution.toRef(), mToLight};
}
