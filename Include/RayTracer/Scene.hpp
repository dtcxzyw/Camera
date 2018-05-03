#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <vector>

class MaterialRef;
class BvhForTriangleRef;
struct Interaction;
struct LightWrapper;

//triangles only

class Primitive final {
private:
    Transform mTrans;
    BvhForTriangleRef* mGeometry;
    MaterialRef* mMaterial;
public:
    Primitive(const Transform& trans, BvhForTriangleRef* geometry, MaterialRef* material);
    CUDA bool intersect(const Ray& ray) const;
    CUDA bool intersect(const Ray& ray, float& tHit, Interaction& interaction) const;
};

class SceneRef final {
private:
    READONLY(Primitive) mPrimitives;
    unsigned int mPrimitiveSize;
public:
    SceneRef(Primitive* primitives, unsigned int priSize);
    CUDA bool intersect(const Ray& ray) const;
    CUDA bool intersect(const Ray& ray, Interaction& interaction) const;
};

class SceneDesc final {
private:
    MemorySpan<Primitive> mPrimitives;
    MemorySpan<LightWrapper*> mLights;
public:
    explicit SceneDesc(const std::vector<Primitive>& priData, 
        const std::vector<LightWrapper*>& lightData);
    SceneRef getRef() const;
};