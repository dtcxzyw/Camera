#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <vector>

class MaterialWrapper;
class BvhForTriangleRef;
struct Interaction;
struct LightWrapper;

class Primitive final {
private:
    Transform mToObject;
    BvhForTriangleRef* mGeometry;
    MaterialWrapper* mMaterial;
public:
    Primitive(const Transform& trans, BvhForTriangleRef* geometry, MaterialWrapper* material);
    DEVICE bool intersect(const Ray& ray) const;
    DEVICE bool intersect(const Ray& ray, float& tHit, Interaction& interaction) const;
};

class SceneRef final {
private:
    READONLY(Primitive) mPrimitives;
    uint32_t mPrimitiveSize;
    LightWrapper** mLights;
    uint32_t mLightSize;
public:
    SceneRef(Primitive* primitives, uint32_t priSize, LightWrapper** light, uint32_t lightSize);
    DEVICE bool intersect(const Ray& ray) const;
    DEVICE bool intersect(const Ray& ray, Interaction& interaction) const;
    DEVICE LightWrapper** begin() const;
    DEVICE LightWrapper** end() const;
    DEVICE uint32_t size() const;
};

class SceneDesc final {
private:
    MemorySpan<Primitive> mPrimitives;
    MemorySpan<LightWrapper*> mLights;
public:
    explicit SceneDesc(const std::vector<Primitive>& priData,
        const std::vector<LightWrapper*>& lightData);
    SceneRef toRef() const;
};
