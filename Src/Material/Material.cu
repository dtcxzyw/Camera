#include <Material/Material.hpp>

DEVICE Vector Bsdf::toLocal(const Vector& vec) const {
    auto&& shading = mInteraction.shadingGeometry;
    return {dot(vec, shading.dpdu), dot(vec, shading.dpdv), dot(vec, Vector{shading.normal})};
}

DEVICE Vector Bsdf::toWorld(const Vector& vec) const {
    auto&& shading = mInteraction.shadingGeometry;
    return mat3(shading.dpdu, shading.dpdv, Vector(shading.normal)) * vec;
}

DEVICE float Bsdf::pdfImpl(const Vector& wo, const Vector& wi, const BxDFType pattern) const {
    if (wo.z == 0.0f) return 0.0f;
    auto pdf = 0.0f;
    auto count = 0U;
    for (auto i = 0U; i < mCount; ++i)
        if (mBxDF[i].match(pattern)) {
            ++count;
            pdf += mBxDF[i].pdf(wo, wi);
        }
    return count ? pdf / mCount : 0.0f;
}

DEVICE Spectrum Bsdf::fImpl(const Vector& wo, const Vector& worldWo, const Vector& wi, const Vector& worldWi,
    const BxDFType pattern) const {
    if (wo.z == 0.0f) return Spectrum{};
    auto&& shading = mInteraction.shadingGeometry;
    const auto flags = pattern | (dot(worldWo, Vector{shading.normal})
                                  * dot(worldWi, Vector(shading.normal)) > 0.0f
                                      ? BxDFType::Reflection
                                      : BxDFType::Transmission);
    Spectrum f{};
    for (auto i = 0U; i < mCount; ++i)
        if (mBxDF[i].match(flags))
            f += mBxDF[i].f(wo, wi);
    return f;
}

DEVICE Bsdf::Bsdf(const Interaction& interaction) : mInteraction(interaction), mCount(0), mEta(1.0f) {}

DEVICE unsigned int Bsdf::match(const BxDFType type) const {
    auto res = 0U;
    for (auto i = 0U; i < mCount; ++i)
        res += mBxDF[i].match(type);
    return res;
}

DEVICE float Bsdf::pdf(const Vector& worldWo, const Vector& worldWi, const BxDFType pattern) const {
    return pdfImpl(toLocal(worldWo), toLocal(worldWi), pattern);
}

DEVICE Spectrum Bsdf::f(const Vector& worldWo, const Vector& worldWi, const BxDFType pattern) const {
    return fImpl(toLocal(worldWo), worldWo, toLocal(worldWi), worldWi, pattern);
}

DEVICE BxDFSample Bsdf::sampleF(const Vector& worldWo, const vec2 sample, const BxDFType pattern) const {
    const auto count = match(pattern);
    if (count == 0) return {};
    const auto nth = min(static_cast<unsigned int>(sample.x * count), count - 1U);

    auto id = 0U;
    for (auto i = 0U, cur = 0U; i < mCount; ++i)
        if (mBxDF[i].match(pattern) && cur++ == nth) {
            id = i;
            break;
        }

    const vec2 sampleRemapped(fmin(sample.x * count - nth, 1.0f), sample.y);

    const auto wo = toLocal(worldWo);
    if (wo.z == 0.0f) return {};
    const auto res = mBxDF[id].sampleF(wo, sampleRemapped);
    if (res.pdf == 0.0f)return {};
    const auto wi = toWorld(Vector(res.wi));
    if (mBxDF[id].match(BxDFType::Specular))return {wi, res.f, res.type, res.pdf / count};
    return {
        wi, fImpl(wo, worldWo, Vector{res.wi}, wi, pattern), res.type,
        pdfImpl(wo, Vector{res.wi}, pattern) / count
    };
}
