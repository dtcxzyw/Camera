#include <Sampler/Sampling.hpp>
#include <IO/Image.hpp>

BOTH float computeCdf(const float* func, float* cdf, const uint32_t size) {
    cdf[0] = 0.0f;
    auto fac = 1.0f / size;
    for (auto i = 0U; i < size; ++i)
        cdf[i + 1] = cdf[i] + func[i] * fac;
    const auto sum = cdf[size];
    if (sum != 0.0f) fac = 1.0f / sum;
    for (auto i = 1U; i <= size; ++i)
        cdf[i] *= fac;
    return sum;
}

static std::pair<MemorySpan<float>, float> computeCdf(const float* val, const uint32_t size) {
    PinnedBuffer<float> cdf(size + 1);
    const auto sum = computeCdf(val, cdf.begin(), size);
    return std::make_pair(upload(cdf), sum);
}

static DEVICE int find(const float* array, const int size, const float x) {
    auto l = 0, r = size - 1, res = -1;
    while (l <= r) {
        const auto mid = (l + r) >> 1;
        if (array[mid] <= x)r = mid - 1, res = mid;
        else l = mid + 1;
    }
    return res;
}

BOTH Distribution1DRef::Distribution1DRef(const float* cdf, const float* func, const uint32_t size, const float sum)
    : mCdf(cdf), mFunc(func), mSize(size), mInvLength(1.0f / (size - 1)), mInvSum(sum ? 1.0f / sum : 0.0f) {}

DEVICE float Distribution1DRef::sampleContinuous(const float sample, float& pdf, int& pos) const {
    pos = find(mCdf, mSize, sample);
    const auto lv = mCdf[pos], rv = mCdf[pos + 1];
    auto rem = sample - lv;
    const auto delta = rv - lv;
    if (delta)rem /= delta;
    pdf = mFunc[pos] * mInvSum;
    return (pos + rem) * mInvLength;
}

DEVICE int Distribution1DRef::sampleDiscrete(const float sample, float& pdf) const {
    const auto pos = find(mCdf, mSize, sample);
    pdf = mFunc[pos] * mInvSum * mInvLength;
    return pos;
}

DEVICE float Distribution1DRef::f(const uint32_t pos) const {
    return mFunc[pos];
}

DEVICE float Distribution1DRef::getInvSum() const {
    return mInvSum;
}

Distribution1D::Distribution1D(const float* val, const uint32_t size) : mFunc(size) {
    checkError(cudaMemcpy(mFunc.begin(), val, sizeof(float) * size, cudaMemcpyHostToDevice));
    const auto res = computeCdf(val, size);
    mCdf = res.first;
    mSum = res.second;
}

Distribution1DRef Distribution1D::toRef() const {
    return Distribution1DRef{
        mCdf.begin(), mFunc.begin(), static_cast<uint32_t>(mCdf.size()), mSum
    };
}

float Distribution1D::getSum() const {
    return mSum;
}

Distribution2DRef::Distribution2DRef(const Distribution1DRef* refs, const uvec2 size)
    : mRefs(refs), mLines(refs + size.y), mSize(size) {}

DEVICE vec2 Distribution2DRef::sampleContinuous(const vec2 sample, float& pdf) const {
    float pdfX, pdfY;
    int pos;
    const auto y = mLines->sampleContinuous(sample.y, pdfY, pos);
    const auto x = mRefs[pos].sampleContinuous(sample.x, pdfX, pos);
    pdf = pdfX * pdfY;
    return {x, y};
}

DEVICE float Distribution2DRef::pdf(const vec2 sample) const {
    const auto x = min(static_cast<uint32_t>(sample.x * mSize.x), mSize.x - 1);
    const auto y = min(static_cast<uint32_t>(sample.y * mSize.y), mSize.y - 1);
    return mRefs[y].f(x) * mLines->getInvSum();
}

Distribution2D::Distribution2D(const std::string& path) {
    const auto res = loadDistribution2D(path);
    const auto func = std::move(res.first);
    mSize = res.second;
    std::vector<Distribution1DRef> refs;
    std::vector<float> sums;
    refs.reserve(mSize.y);
    sums.reserve(mSize.y);
    mDistribution.reserve(mSize.y);
    for (auto i = 0; i < mSize.y; ++i) {
        mDistribution.emplace_back(func.data() + i * mSize.x, mSize.x);
        refs.emplace_back(mDistribution.back().toRef());
        sums.emplace_back(mDistribution.back().getSum());
    }
    mDistribution.emplace_back(sums.data(), static_cast<uint32_t>(sums.size()));
    refs.emplace_back(mDistribution.back().toRef());
    mRefs = upload(refs);
}

Distribution2DRef Distribution2D::toRef() const {
    return {mRefs.begin(), mSize};
}
