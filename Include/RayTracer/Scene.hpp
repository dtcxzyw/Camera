#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <vector>

class MaterialWrapper;
class BvhForTriangleRef;
struct SurfaceInteraction;
class LightWrapper;
class LightDistributionCache;
class LightDistributionCacheRef;
struct LightDistribution;

class Primitive final {
private:
    Transform mToObject;
    BvhForTriangleRef* mGeometry;
    MaterialWrapper* mMaterial;
public:
    Primitive(const Transform& trans, BvhForTriangleRef* geometry, MaterialWrapper* material);
    DEVICE bool intersect(const Ray& ray) const;
    DEVICE bool intersect(const Ray& ray, float& tHit, SurfaceInteraction& interaction,
        Transform& toWorld) const;
};

class SceneRef final {
private:
    READONLY(Primitive) mPrimitives;
    uint32_t mPrimitiveSize;
    LightWrapper* mLights;
    uint32_t mLightSize;
    const LightDistributionCacheRef* mDistribution;
public:
    SceneRef(Primitive* primitives, uint32_t priSize, LightWrapper* light, uint32_t lightSize,
        const LightDistributionCacheRef* distribution);
    DEVICE bool intersect(const Ray& ray) const;
    DEVICE bool intersect(const Ray& ray, SurfaceInteraction& interaction) const;
    DEVICE LightWrapper* begin() const;
    DEVICE LightWrapper* end() const;
    DEVICE LightWrapper& operator[](uint32_t id) const;
    DEVICE uint32_t size() const;
    DEVICE const LightDistribution* lookUp(const Point& pos) const;
};

class SceneDesc final {
private:
    MemorySpan<Primitive> mPrimitives;
    MemorySpan<LightWrapper> mLights;
    std::unique_ptr<LightDistributionCache> mDistribution;
    MemorySpan<LightDistributionCacheRef> mDistributionRef;
    Bounds mBounds;
public:
    explicit SceneDesc(const std::vector<Primitive>& priData,
        const std::vector<LightWrapper>& lightData, const Bounds& bounds,
        unsigned int lightDistributionVoxels = 0);
    SceneRef toRef() const;
    ~SceneDesc();
};
