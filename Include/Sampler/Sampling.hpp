#pragma once
#include <Math/Math.hpp>
#include <string>
#include <Core/Memory.hpp>
#include <vector>

DEVICEINLINE vec2 concentricSampleDisk(vec2 p) {
    p = 2.0f * p - 1.0f;
    if (p.x == 0.0f & p.y == 0.0f)return {0.0f, 0.0f};
    float r, theta;
    if (fabs(p.x) > fabs(p.y)) {
        r = p.x;
        theta = quarter_pi<float>() * p.y / p.x;
    }
    else {
        r = p.y;
        theta = half_pi<float>() - quarter_pi<float>() * p.x / p.y;
    }
    return r * vec2{cos(theta), sin(theta)};
}

DEVICEINLINE Vector cosineSampleHemisphere(const vec2 p) {
    const auto d = concentricSampleDisk(p);
    return {d.x, d.y, sqrt(1.0f - d.x * d.x - d.y * d.y)};
}

BOTH float computeCdf(const float* func, float* cdf, uint32_t size);

class Distribution1DRef final {
private:
    READONLY(float) mCdf;
    READONLY(float) mFunc;
    uint32_t mSize;
    float mInvLength, mInvSum;
public:
    Distribution1DRef() = default;
    BOTH Distribution1DRef(const float* cdf, const float* func, uint32_t size, float sum);
    DEVICE float sampleContinuous(float sample, float& pdf, int& pos) const;
    DEVICE int sampleDiscrete(float sample, float& pdf) const;
    DEVICE float f(uint32_t pos) const;
    DEVICE float getInvSum() const;
};

class Distribution1D final {
private:
    MemorySpan<float> mCdf, mFunc;
    float mSum;
public:
    Distribution1D(const float* val, uint32_t size);
    Distribution1DRef toRef() const;
    float getSum() const;
};

class Distribution2DRef final {
private:
    const Distribution1DRef* mRefs;
    const Distribution1DRef* mLines;
    uvec2 mSize;
public:
    Distribution2DRef(const Distribution1DRef* refs, uvec2 size);
    DEVICE vec2 sampleContinuous(vec2 sample, float& pdf) const;
    DEVICE float pdf(vec2 sample) const;
};

class Distribution2D final {
private:
    std::vector<Distribution1D> mDistribution;
    MemorySpan<Distribution1DRef> mRefs;
    uvec2 mSize;
public:
    explicit Distribution2D(const std::string& path);
    Distribution2DRef toRef() const;
};
