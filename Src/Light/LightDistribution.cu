#include <Light/LightDistribution.hpp>
#include <Core/DeviceFunctions.hpp>
//TODO:LightGroup
#include <RayTracer/Scene.hpp>
#include <Sampler/SequenceGenerator.hpp>
#include <Light/LightWrapper.hpp>

DEVICE void BoundsDistribution::computeDistribution(const SceneRef& scene,
    const Bounds& bounds, float* buf) {
    const auto func = buf;
    constexpr auto sampleNum = 128U;
    for (auto i = 0; i < sampleNum; ++i) {
        Interaction isect;
        isect.pos = bounds.lerp({radicalInverse2(i), halton3(i), halton5(i)});
        const vec2 sample = {halton7(i), halton11(i)};
        for (auto j = 0; j < scene.size(); ++j) {
            const auto ls = scene[j].sampleLi(sample, isect);

            if (ls.pdf > 0.0f) {
                const auto lum = ls.illumination.y();
                if (lum > 0.0f) {
                    if (!scene.intersect(isect.spawnTo(ls.src)))
                        func[j] += lum / ls.pdf;
                }
            }
        }
    }

    auto acc = 0.0f;
    for (auto i = 0; i < scene.size(); ++i)
        acc += func[i];
    const auto minv = acc == 0.0f ? 1.0f : acc / (scene.size() * sampleNum) * 1e-3f;
    for (auto i = 0; i < scene.size(); ++i)
        func[i] = fmax(func[i], minv);

    const auto cdf = buf + scene.size();
    const auto sum = computeCdf(func, cdf, scene.size());
    distribution = Distribution1DRef{cdf, func, scene.size() + 1, sum};
    fIag = 0;
}

DEVICE int BoundsDistribution::chooseOneLight(const float sample, float& pdf) const {
    return distribution.sampleDiscrete(sample, pdf);
}

DEVICE void PointDistribution::computeDistribudion(const SceneRef& scene, const Point& pos,
    float* buf) {
    constexpr auto sampleNum = 128U;
    pdfs = buf;
    Interaction isect;
    isect.pos = pos;
    for (auto i = 0; i < sampleNum; ++i) {
        const vec2 sample = { radicalInverse2(i), halton3(i) };
        for (auto j = 0; j < scene.size(); ++j) {
            const auto ls = scene[j].sampleLi(sample, isect);

            if (ls.pdf > 0.0f) {
                const auto lum = ls.illumination.y();
                if (lum > 0.0f) {
                    if (!scene.intersect(isect.spawnTo(ls.src)))
                        pdfs[j] += lum / ls.pdf;
                }
            }
        }
    }

    auto acc = 0.0f;
    for (auto i = 0; i < scene.size(); ++i)
        acc += pdfs[i];
    const auto minv = acc == 0.0f ? 1.0f : acc / (scene.size() * sampleNum) * 1e-3f;
    for (auto i = 0; i < scene.size(); ++i)
        pdfs[i] = fmax(pdfs[i], minv);

    acc = 0.0f;
    for (auto i = 0; i < scene.size(); ++i)
        acc += pdfs[i];

    const auto fac = 1.0f / (acc*scene.size());
    for (auto i = 0; i < scene.size(); ++i)
        pdfs[i] *= fac;
}

DEVICE float PointDistribution::pdf(const int id) const {
    return pdfs[id];
}

int LightDistribution::chooseOneLight(const float sample, float& pdf) const {
    if (boundsDistribution) {
        const auto id = boundsDistribution->chooseOneLight(sample, pdf);
        pdf = 0.0f;
        for (auto i = 0U; i < 8U; ++i) 
            pdf += pointDistribution[i]->pdf(id)*weight[i];
        return id;
    }
    //TODO:Power Distribution
    //Uniform Distribution
    {
        pdf = invSize;
        return min(static_cast<unsigned int>(sample*size), end);
    }
}

DEVICE uint32_t encode(const uint64_t packedPos, const uint32_t size) {
    auto hash = packedPos;
    hash ^= (hash >> 31);
    hash *= 0x7fb5d329728ea185;
    hash ^= (hash >> 27);
    hash *= 0x81dadef4bc2dd44d;
    hash ^= (hash >> 33);
    return hash % size;
}

const BoundsDistribution* LightDistributionCacheRef::lookUpBounds(const SceneRef& scene, 
    const uvec3& posi) const {
    const auto packedPos = static_cast<uint64_t>(posi.x) << 40 | static_cast<uint64_t>(posi.y) << 20 | posi.z;
    auto id = encode(packedPos, mBoundsSize);
    auto step = 1U;
    while (true) {
        auto&& dist = mBoundsDistributions[id];
        if (dist.packedPos == packedPos) {
            if (dist.fIag)continue; //for avoiding to deadlock
            return &dist;
        }
        if (deviceCompareAndSwap(&dist.packedPos, static_cast<uint64_t>(-1), packedPos) == -1) {
            const auto p0 = mBase + static_cast<Vector>(posi) * mScale;
            const auto p1 = p0 + mScale;
            dist.computeDistribution(scene, Bounds{ p0, p1 },
                mBuffer + deviceAtomicAdd(&mPoolOffset, static_cast<uint64_t>(scene.size() * 2 + 1)));
            return &dist;
        }
        {
            id = (id + step * step) % mBoundsSize;
            ++step;
        }
    }
}

DEVICE const PointDistribution * LightDistributionCacheRef::lookUpPoint(const SceneRef & scene, const uvec3 & posi) const{
    const auto packedPos = static_cast<uint64_t>(posi.x) << 40 | static_cast<uint64_t>(posi.y) << 20 | posi.z;
    auto id = encode(packedPos, mPointsSize);
    auto step = 1U;
    while (true) {
        auto&& dist = mPointDistributions[id];
        if (dist.packedPos == packedPos) {
            if (dist.flag)continue; //for avoiding to deadlock
            return &dist;
        }
        if (deviceCompareAndSwap(&dist.packedPos, static_cast<uint64_t>(-1), packedPos) == -1) {
            dist.computeDistribudion(scene, mBase + static_cast<Vector>(posi) * mScale,
                mBuffer + deviceAtomicAdd(&mPoolOffset, static_cast<uint64_t>(scene.size())));
            return &dist;
        }
        {
            id = (id + step * step) % mPointsSize;
            ++step;
        }
    }
}

LightDistributionCacheRef::LightDistributionCacheRef(BoundsDistribution* boundsPtr,
    const uint32_t boundsSize, PointDistribution*pointsPtr, const uint32_t pointsSize,
    const uvec3 size, const Bounds& bounds, float* buffer)
    : mBoundsDistributions(boundsPtr), mPointDistributions(pointsPtr), mSize(size),
    mBase(bounds[0]), mScale((bounds[1] - bounds[0]) / static_cast<Vector>(size)),
    mInvScale(static_cast<Vector>(size) / (bounds[1] - bounds[0])), mBoundsSize(boundsSize),
    mPointsSize(pointsSize), mPoolOffset(0), mBuffer(buffer) {}

DEVICE LightDistribution LightDistributionCacheRef::lookUp(const SceneRef& scene,
    const Point& worldPos) const {
    const auto pos = clamp((worldPos - mBase) * mScale, Vector{ 0.0f }, Vector{ mSize } -1e-5f);
    const uvec3 posi = pos;
    LightDistribution distribution;
    /*distribution.boundsDistribution = lookUpBounds(scene, posi);
    for (auto i = 0U; i < 8U; ++i) {
        const auto p = posi + uvec3{ i & 1,(i & 2) >> 1,(i & 4) >> 2 };
        distribution.pointDistribution[i] = lookUpPoint(scene, p);
        const auto off = Vector{ p } -pos;
        distribution.weight[i] = fabs(off.x*off.y*off.z);
    }
    */
    return distribution;
}

LightDistributionCache::LightDistributionCache(const Bounds& bounds,
    const uint32_t maxVoxels, const size_t lightSize) : mBounds(bounds) {
    const auto offset = bounds[1] - bounds[0];
    const auto maxOffset = offset[maxDim(offset)];
    for (auto i = 0; i < 3; ++i) {
        const uint32_t size = round(offset[i] / maxOffset * maxVoxels);
        mSize[i] = max(size, 1U);
    }
    mBoundsDistributions = MemorySpan<BoundsDistribution>(mSize.x * mSize.y * mSize.z);
    mBoundsDistributions.memset(0xff);
    mPointDistributions = MemorySpan<PointDistribution>((mSize.x + 1) * (mSize.y + 1) * (mSize.z + 1));
    mMemoryPool = MemorySpan<float>((lightSize * 2 + 1)*mBoundsDistributions.size() +
        lightSize * mPointDistributions.size());
    mMemoryPool.memset(0);
}

LightDistributionCacheRef LightDistributionCache::toRef() const {
    return {
        mBoundsDistributions.begin(), static_cast<uint32_t>(mBoundsDistributions.size()),
        mPointDistributions.begin(),static_cast<uint32_t>(mPointDistributions.size()),
        mSize,mBounds,mMemoryPool.begin()
    };
}
