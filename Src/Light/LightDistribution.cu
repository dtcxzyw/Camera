#include <Light/LightDistribution.hpp>
#include <Core/DeviceFunctions.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>
//TODO:LightGroup
#include <RayTracer/Scene.hpp>
#include <Sampler/SequenceGenerator.hpp>
#include <Light/LightWrapper.hpp>

DEVICE void LightDistribution::computeDistribution(const SceneRef& scene, const Bounds& bounds) {
    func = new float[scene.size()];
    constexpr auto sampleNum = 64U;
    for (auto i = 0; i < sampleNum; ++i) {
        const auto pos = bounds.lerp({ radicalInverse2(i), halton3(i), halton5(i) });
        const vec2 sample = { halton7(i),halton11(i) };
        for (auto j = 0; j < scene.size(); ++j) {
            const auto ls = scene.begin()[j]->sampleLi(sample, pos);
            if (ls.pdf > 0.0f) {
                const auto lum = ls.illumination.lum();
                /*
                 *TODO:trace shadow rays
                if (lum > 0.0f) {
                    if (!scene.intersect(Ray{ pos ,ls.src-pos}))
                        func[j] += lum / ls.pdf;
                }
                */
                func[j] += lum / ls.pdf;
            }
        }
    }

    auto acc = 0.0f;
    for (auto i = 0; i < scene.size(); ++i)
        acc += func[i];
    const auto minv = acc == 0.0f ? 1.0f : acc / (scene.size()*sampleNum)*1e-3f;
    for (auto i = 0; i < scene.size(); ++i)
        func[i] = fmax(func[i], minv);

    cdf = new float[scene.size() + 1];
    const auto sum = computeCdf(func, cdf, scene.size());
    distribution = Distribution1DRef(cdf, func, scene.size() + 1, sum);
    fIag = 0;
}

DEVICE int LightDistribution::chooseOneLight(const float sample,float& pdf) const {
    return distribution.sampleDiscrete(sample, pdf);
}

LightDistributionCacheRef::LightDistributionCacheRef(LightDistribution* distribution,
    const uint32_t cacheSize, const uvec3 size, const Bounds& bounds)
    : mDistributions(distribution), mCacheSize(cacheSize), mSize(size), 
    mBase(bounds[0]), mInvOffset(1.0f / (bounds[1] - bounds[0])),
    mScale((bounds[1] - bounds[0]) / static_cast<Vector>(size)) {}

DEVICE uint64_t encode(const uint64_t packedPos, const uint32_t size) {
    auto hash = packedPos;
    hash ^= (hash >> 31);
    hash *= 0x7fb5d329728ea185;
    hash ^= (hash >> 27);
    hash *= 0x81dadef4bc2dd44d;
    hash ^= (hash >> 33);
    return hash % size;
}

DEVICE const LightDistribution* LightDistributionCacheRef::lookUp(const SceneRef& scene, 
    const Point& worldPos) const {
    const auto pos = (worldPos - mBase)*mInvOffset;
    uvec3 posi;
    #pragma unroll
    for (auto i = 0; i < 3; ++i)
        posi[i] = clamp(static_cast<int>(pos[i] * mSize[i]), 0, static_cast<int>(mSize[i] - 1));
    const auto packedPos = static_cast<uint64_t>(posi.x) << 40 | static_cast<uint64_t>(posi.y) << 20 | posi.z;
    auto id = encode(packedPos, mCacheSize);
    auto step = 1U;
    for (auto i = 0; i < 4; ++i) {
        auto&& dist = mDistributions[id];
        if (dist.packedPos == packedPos) {
            if (dist.fIag)return nullptr;
            //For avoiding to deadlock with threads in the same warp, we return nullptr.
            return &dist;
        }
        if (deviceCompareAndSwap(&dist.packedPos, static_cast<uint64_t>(-1), packedPos) == -1) {
            const auto p0 = mBase + static_cast<Vector>(posi)*mScale;
            const auto p1 = p0 + mScale;
            dist.computeDistribution(scene, Bounds(p0, p1));
            return &dist;
        }
        {
            id = (id + step * step) % mCacheSize;
            ++step;
        }
    }
    return nullptr;
}

LightDistributionCache::LightDistributionCache(const Bounds& bounds,
    const uint32_t maxVoxels) : mBounds(bounds) {
    const auto offset = bounds[1] - bounds[0];
    const auto maxOffset = offset[maxDim(offset)];
    for (auto i = 0; i < 3; ++i) {
        const uint32_t size = round(offset[i] / maxOffset * maxVoxels);
        mSize[i] = max(size, 1U);
    }
    mDistributions = MemorySpan<LightDistribution>(4 * mSize.x * mSize.y * mSize.z);
    mDistributions.memset(0xff);
}

static GLOBAL void destoryDistribution(const uint32_t size, LightDistribution* distribution) {
    const auto id = getId();
    if (id<size & distribution[id].fIag == 0) {
        delete[] distribution[id].func;
        delete[] distribution[id].cdf;
    }
}

LightDistributionCache::~LightDistributionCache() {
    auto buffer = std::make_unique<CommandBuffer>();
    buffer->launchKernelLinear(makeKernelDesc(destoryDistribution), mDistributions.size(),
        buffer->useAllocated(mDistributions));
    Environment::get().submit(std::move(buffer)).sync();
}

LightDistributionCacheRef LightDistributionCache::toRef() const {
    return { mDistributions.begin() ,static_cast<uint32_t>(mDistributions.size()), mSize,mBounds };
}
