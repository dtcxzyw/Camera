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
        interaction.toWorld = inverse(mTrans);
        interaction.material = mMaterial;
        return true;
    }
    return false;
}

SceneRef::SceneRef(Primitive* primitives, const unsigned int priSize)
    : mPrimitives(primitives), mPrimitiveSize(priSize) {}

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

SceneDesc::SceneDesc(const std::vector<Primitive>& priData, const std::vector<LightWrapper*>& light) 
    :mPrimitives(upload(priData)), mLights(upload(light)) {}

SceneRef SceneDesc::getRef() const {
    return SceneRef(mPrimitives.begin(), mPrimitives.size());
}
