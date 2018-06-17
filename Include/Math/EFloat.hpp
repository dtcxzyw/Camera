#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

inline BOTH float gamma(const int n) {
    const auto x = n * (epsilon<float>() * 0.5f);
    return x / (1.0f - x);
}

inline BOTH float nextFloatDown(float v) {
    if (isinf(v) & (v < 0.0f)) return v;
    if (v == 0.0f) v = -0.0f;
    auto ui = binCast<uint32_t>(v);
    if (v > 0.0f)--ui;
    else ++ui;
    return binCast<float>(ui);
}

inline BOTH float nextFloatUp(float v) {
    if (isinf(v) & (v > 0.0f)) return v;
    if (v == -0.0f) v = 0.0f;
    auto ui = binCast<uint32_t>(v);
    if (v >= 0.0f)++ui;
    else --ui;
    return binCast<float>(ui);
}

class EFloat final {
private:
    float mLow, mHigh;
public:
    EFloat() = default;
    BOTH    explicit EFloat(const float val) : mLow(val), mHigh(val) {}
    BOTH    EFloat(const float low, const float high): mLow(low), mHigh(high) {}
    BOTH    EFloat operator+(const EFloat rhs) const {
        return {
            nextFloatDown(lowerBound() + rhs.lowerBound()),
            nextFloatUp(upperBound() + rhs.upperBound())
        };
    }

    BOTH    EFloat& operator+=(const EFloat rhs) {
        return *this = *this + rhs;
    }

    BOTH    EFloat operator-(const EFloat rhs) const {
        return {
            nextFloatDown(lowerBound() - rhs.upperBound()),
            nextFloatUp(upperBound() - rhs.lowerBound())
        };
    }

    BOTH    EFloat& operator-=(const EFloat rhs) {
        return *this = *this - rhs;
    }

    BOTH    EFloat operator-() const {
        return {-upperBound(), -lowerBound()};
    }

    BOTH    EFloat operator*(const EFloat rhs) const {
        const float val[] = {
            lowerBound() * rhs.lowerBound(), lowerBound() * rhs.upperBound(),
            upperBound() * rhs.lowerBound(), upperBound() * rhs.lowerBound()
        };
        return {
            nextFloatDown(fmin(fmin(val[0], val[1]), fmin(val[2], val[3]))), nextFloatUp(
                fmax(fmax(val[0], val[1]), fmax(val[2], val[3])))
        };
    }

    BOTH    EFloat& operator*=(const EFloat rhs) {
        return *this = *this * rhs;
    }

    BOTH    EFloat operator/(const EFloat rhs) const {
        const float val[] = {
            lowerBound() / rhs.lowerBound(), lowerBound() / rhs.upperBound(),
            upperBound() / rhs.lowerBound(), upperBound() / rhs.lowerBound()
        };
        return {
            nextFloatDown(fmin(fmin(val[0], val[1]), fmin(val[2], val[3]))), nextFloatUp(
                fmax(fmax(val[0], val[1]), fmax(val[2], val[3])))
        };
    }

    BOTH    EFloat& operator/=(const EFloat rhs) {
        return *this = *this / rhs;
    }

    BOTH    float lowerBound() const {
        return mLow;
    }

    BOTH    float upperBound() const {
        return mHigh;
    }
};

inline BOTH EFloat makeEFloat(const float val, const float err = 0.0f) {
    return err == 0.0f ? EFloat{ val } : EFloat{ nextFloatDown(val - err), nextFloatUp(val + err) };
}

inline BOTH EFloat sqrt(const EFloat val) {
    return {nextFloatDown(val.lowerBound()), nextFloatUp(val.upperBound())};
}

inline BOTH EFloat abs(const EFloat val) {
    if (val.lowerBound() >= 0.0f)return val;
    if (val.upperBound() >= 0.0f)return {0.0f, fmax(-val.lowerBound(), val.upperBound())};
    return {-val.upperBound(), -val.lowerBound()};
}

DEVICEINLINE bool quadratic(const EFloat a, const EFloat b, const EFloat c, EFloat& x1, EFloat& x2) {
    const auto discrim = b * b - EFloat{ 4.0f } *a * c;
    if (discrim.lowerBound() < 0.0f) return false;
    const auto rootDiscrim = sqrt(discrim);

    const auto q = EFloat{ -0.5f } *((b.lowerBound() < 0.0f) ? b - rootDiscrim : b + rootDiscrim);
    x1 = q / a;
    x2 = c / q;
    if (x1.lowerBound() > x2.lowerBound())cudaSwap(x1, x2);
    return true;
}
