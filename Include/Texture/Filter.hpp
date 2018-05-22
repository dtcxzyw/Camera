#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

struct FilterTag {};

class BoxFilter final : FilterTag {
public:
    DEVICE float evaluate(const vec2& pos) const {
        return 1.0f;
    }
};

class TriangleFilter final : FilterTag {
public:
    DEVICE float evaluate(const vec2& pos) const {
        return (1.0f - pos.x) * (1.0f - pos.y);
    }
};

class GaussianFilter final : FilterTag {
private:
    const float mExpv;
public:
    explicit GaussianFilter(const float alpha) : mExpv(expf(-alpha)) {}
    DEVICE float evaluate(const vec2& pos) const {
        return powf(mExpv, pos.x * pos.x) * powf(mExpv, pos.y * pos.y);
    }
};

DEVICEINLINE float sinc(const float x) {
    if (x < 1e-5f)return 1.0f;
    const auto px = pi<float>() * x;
    return sin(px) / px;
}

DEVICEINLINE float windowedSinc(const float x, const float invTau) {
    return sinc(x) * sinc(x * invTau);
}

class LanczosSincFilter final : FilterTag {
private:
    const float mInvTau;
public:
    explicit LanczosSincFilter(const float tau) : mInvTau(1.0f / tau) {}
    DEVICE float evaluate(const vec2& pos) const {
        return windowedSinc(pos.x, mInvTau) * windowedSinc(pos.y, mInvTau);
    }
};
