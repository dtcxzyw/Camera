#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <Sampler/Sampling.hpp>

class SceneRef;

struct BoundsDistribution final {
    Distribution1DRef distribution;
    uint64_t packedPos;
    int fIag;
    DEVICE void computeDistribution(const SceneRef& scene, const Bounds& posi, float* buf);
    DEVICE int chooseOneLight(float sample,float& pdf) const;
};

struct PointDistribution final {
    uint64_t packedPos;
    float* pdfs;
    int flag;
    DEVICE void computeDistribudion(const SceneRef& scene, const Point& pos, float* buf);
    DEVICE float pdf(int id) const;
};

struct LightDistribution final {
    const BoundsDistribution* boundsDistribution{};
    const PointDistribution* pointDistribution[8]{};
    float invSize, size, weight[8]{};
    uint32_t end;
    DEVICE int chooseOneLight(float sample, float& pdf) const;
};

class LightDistributionCacheRef final {
private:
    BoundsDistribution* mBoundsDistributions;
    PointDistribution* mPointDistributions;
    uvec3 mSize;
    Point mBase;
    Vector mScale, mInvScale;
    uint32_t mBoundsSize, mPointsSize;
    mutable uint64_t mPoolOffset;
    float* mBuffer;
    DEVICE const BoundsDistribution* lookUpBounds(const SceneRef& scene, const uvec3& posi) const;
    DEVICE const PointDistribution* lookUpPoint(const SceneRef& scene, const uvec3& posi) const;
public:
    LightDistributionCacheRef(BoundsDistribution* boundsPtr,
        uint32_t boundsSize, PointDistribution*pointsPtr, uint32_t pointsSize,
        uvec3 size, const Bounds& bounds, float* buffer);
    DEVICE LightDistribution lookUp(const SceneRef& scene, const Point& worldPos) const;
};

class LightDistributionCache final : Uncopyable {
private:
    MemorySpan<BoundsDistribution> mBoundsDistributions;
    MemorySpan<PointDistribution> mPointDistributions;
    MemorySpan<float> mMemoryPool;
    uvec3 mSize;
    Bounds mBounds;
public:
    explicit LightDistributionCache(const Bounds& bounds, uint32_t maxVoxels, size_t lightSize);
    LightDistributionCacheRef toRef() const;
};
