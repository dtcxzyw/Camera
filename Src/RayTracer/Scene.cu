#include <RayTracer/Scene.hpp>
#include <RayTracer/BVH.hpp>
#include <Math/Interaction.hpp>
#include <Light/LightWrapper.hpp>

Primitive::Primitive(const Transform& trans, BvhForTriangleRef* geometry, MaterialWrapper* material)
    : mToObject(inverse(trans)), mGeometry(geometry), mMaterial(material) {}

DEVICE bool Primitive::intersect(const Ray& ray) const {
    return mGeometry->intersect(mToObject(ray));
}

DEVICE bool Primitive::intersect(const Ray& ray, float& tHit, Interaction& interaction) const {
    if (mGeometry->intersect(mToObject(ray), tHit, interaction)) {
        interaction.toWorld = inverse(mToObject);
        interaction.material = mMaterial;
        return true;
    }
    return false;
}

SceneRef::SceneRef(Primitive* primitives, const uint32_t priSize,
    LightWrapper** light, const uint32_t lightSize)
    : mPrimitives(primitives), mPrimitiveSize(priSize), mLights(light), mLightSize(lightSize) {}

DEVICE bool SceneRef::intersect(const Ray& ray) const {
    for (auto i = 0U; i < mPrimitiveSize; ++i)
        if (mPrimitives[i].intersect(ray))
            return true;
    return false;
}

DEVICE bool SceneRef::intersect(const Ray& ray, Interaction& interaction) const {
    auto tHit = ray.tMax;
    auto flag = false;
    for (auto i = 0U; i < mPrimitiveSize; ++i)
        flag |= mPrimitives[i].intersect(ray, tHit, interaction);
    return flag;
}

DEVICE LightWrapper** SceneRef::begin() const {
    return mLights;
}

DEVICE LightWrapper** SceneRef::end() const {
    return mLights + mLightSize;
}

DEVICE uint32_t SceneRef::size() const {
    return mLightSize;
}

SceneDesc::SceneDesc(const std::vector<Primitive>& priData, const std::vector<LightWrapper*>& light)
    : mPrimitives(upload(priData)), mLights(upload(light)) {}

SceneRef SceneDesc::toRef() const {
    return SceneRef(mPrimitives.begin(), mPrimitives.size(), mLights.begin(), mLights.size());
}
