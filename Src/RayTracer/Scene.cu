#include <RayTracer/Scene.hpp>
#include <RayTracer/BVH.hpp>
#include <Math/Interaction.hpp>
#include <Light/LightWrapper.hpp>

Primitive::Primitive(const Transform& trans, BvhForTriangleRef* geometry, MaterialWrapper* material)
    : mTrans(trans), mGeometry(geometry), mMaterial(material) {}

DEVICE bool Primitive::intersect(const Ray& ray) const {
    return mGeometry->intersect(mTrans(ray));
}

bool Primitive::intersect(const Ray& ray, float& tHit, Interaction& interaction) const {
    if (mGeometry->intersect(mTrans(ray), tHit, interaction)) {
        interaction.toWorld = inverse(mTrans) * interaction.toWorld;
        interaction.material = mMaterial;
        return true;
    }
    return false;
}

SceneRef::SceneRef(Primitive* primitives, const unsigned int priSize,
    LightWrapper** light, const unsigned int lightSize)
    : mPrimitives(primitives), mPrimitiveSize(priSize), mLights(light), mLightSize(lightSize) {}

bool SceneRef::intersect(const Ray& ray) const {
    for (auto i = 0U; i < mPrimitiveSize; ++i)
        if (mPrimitives[i].intersect(ray))
            return true;
    return false;
}

bool SceneRef::intersect(const Ray& ray, Interaction& interaction) const {
    auto tHit = ray.tMax;
    auto flag = false;
    for (auto i = 0U; i < mPrimitiveSize; ++i)
        flag |= mPrimitives[i].intersect(ray, tHit, interaction);
    return flag;
}

LightWrapper** SceneRef::begin() const {
    return mLights;
}

LightWrapper** SceneRef::end() const {
    return mLights + mLightSize;
}

SceneDesc::SceneDesc(const std::vector<Primitive>& priData, const std::vector<LightWrapper*>& light)
    : mPrimitives(upload(priData)), mLights(upload(light)) {}

SceneRef SceneDesc::toRef() const {
    return SceneRef(mPrimitives.begin(), mPrimitives.size(), mLights.begin(), mLights.size());
}
