#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <vector>

class MaterialWrapper;
class BvhForTriangleRef;
struct Interaction;
struct LightWrapper;

//triangles only

class Primitive final {
private:
    Transform mTrans;
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
    unsigned int mPrimitiveSize;
    LightWrapper** mLights;
    unsigned int mLightSize;
public:
    SceneRef(Primitive* primitives, unsigned int priSize, LightWrapper** light, unsigned int lightSize);
    DEVICE bool intersect(const Ray& ray) const;
    DEVICE bool intersect(const Ray& ray, Interaction& interaction) const;
    DEVICE LightWrapper** begin() const;
    DEVICE LightWrapper** end() const;
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
