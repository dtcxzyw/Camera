#include <RayTracer/Scene.hpp>
#include <RayTracer/BVH.hpp>
#include <Math/Interaction.hpp>
#include <Light/LightWrapper.hpp>
#include <Light/LightDistribution.hpp>

Primitive::Primitive(const Transform& trans, BvhForTriangleRef* geometry, MaterialWrapper* material)
    : mToObject(inverse(trans)), mGeometry(geometry), mMaterial(material) {}

DEVICE bool Primitive::intersect(const Ray& ray) const {
    return mGeometry->intersect(mToObject(ray));
}

DEVICE bool Primitive::intersect(const Ray& ray, float& tHit, SurfaceInteraction& interaction,
    Transform& toWorld) const {
    if (mGeometry->intersect(mToObject(ray), tHit, interaction)) {
        toWorld = inverse(mToObject);
        interaction.material = mMaterial;
        return true;
    }
    return false;
}

SceneRef::SceneRef(Primitive* primitives, const uint32_t priSize,
    LightWrapper* light, const uint32_t lightSize, const LightDistributionCacheRef* distribution)
    : mPrimitives(primitives), mPrimitiveSize(priSize), mLights(light), mLightSize(lightSize),
    mDistribution(distribution) {}

DEVICE bool SceneRef::intersect(const Ray& ray) const {
    for (auto i = 0U; i < mLightSize; ++i)
        if (mLights[i].intersect(ray))
            return true;
    for (auto i = 0U; i < mPrimitiveSize; ++i)
        if (mPrimitives[i].intersect(ray))
            return true;
    return false;
}

DEVICE bool SceneRef::intersect(const Ray& ray, SurfaceInteraction& interaction) const {
    auto tHit = ray.tMax;
    auto flag = false;
    Transform toWorld;
    for (auto i = 0U; i < mLightSize; ++i)
        if(mLights[i].intersect(ray, tHit, interaction)) {
            interaction.areaLight = mLights + i;
            flag = true;
        }
    auto shouldTransform = false;
    for (auto i = 0U; i < mPrimitiveSize; ++i)
        if(mPrimitives[i].intersect(ray, tHit, interaction, toWorld)) {
            interaction.areaLight = nullptr;
            flag = true;
            shouldTransform = true;
        }
    if (shouldTransform)interaction.applyTransform(toWorld);
    return flag;
}

DEVICE LightWrapper* SceneRef::begin() const {
    return mLights;
}

DEVICE LightWrapper* SceneRef::end() const {
    return mLights + mLightSize;
}

DEVICE LightWrapper& SceneRef::operator[](const uint32_t id) const {
    return mLights[id];
}

DEVICE uint32_t SceneRef::size() const {
    return mLightSize;
}

DEVICE const LightDistribution* SceneRef::lookUp(const Point& pos) const {
    return mDistribution ? mDistribution->lookUp(*this, pos) : nullptr;
}

SceneDesc::SceneDesc(const std::vector<Primitive>& priData,
    const std::vector<LightWrapper>& light, const Bounds& bounds,
    const unsigned int lightDistributionVoxels)
    : mPrimitives(upload(priData)), mLights(upload(light)), mBounds(bounds) {
    if (lightDistributionVoxels && light.size() > 1) {
        mDistribution = std::make_unique<LightDistributionCache>(bounds, lightDistributionVoxels);
        LightDistributionCacheRef ref[1] = {mDistribution->toRef()};
        mDistributionRef = upload(ref);
    }
}

SceneRef SceneDesc::toRef() const {
    return {
        mPrimitives.begin(), static_cast<uint32_t>(mPrimitives.size()), mLights.begin(),
        static_cast<uint32_t>(mLights.size()), mDistributionRef.begin()
    };
}

SceneDesc::~SceneDesc() {}
