#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <Sampler/Sampling.hpp>

class SceneRef;

struct LightDistribution final {
    float *cdf, *func;
    Distribution1DRef distribution;
    uint64_t packedPos;
    int fIag;
    DEVICE void computeDistribution(const SceneRef& scene, const Bounds& posi);
    DEVICE int chooseOneLight(float sample, float& pdf) const;
};

class LightDistributionCacheRef final {
private:
    LightDistribution* mDistributions;
    uint32_t mCacheSize;
    uvec3 mSize;
    Point mBase;
    Vector mInvOffset, mScale;
public:
    LightDistributionCacheRef(LightDistribution* distribution, uint32_t cacheSize, uvec3 size, 
        const Bounds& bounds);
    DEVICE const LightDistribution* lookUp(const SceneRef& scene,const Point& worldPos) const;
};

class LightDistributionCache final :Uncopyable {
private:
    MemorySpan<LightDistribution> mDistributions;
    uvec3 mSize;
    Bounds mBounds;
public:
    explicit LightDistributionCache(const Bounds& bounds, uint32_t maxVoxels);
    ~LightDistributionCache();
    LightDistributionCacheRef toRef() const;
};
