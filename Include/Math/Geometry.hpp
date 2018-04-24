#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

using Vector = vec3;

class Point final {
private:
    Vector mPos;
public:
    BOTH Point(const float x, const float y, const float z) : mPos(x, y, z) {}
    BOTH Point& operator+=(const Vector& rhs) {
        mPos += rhs;
        return *this;
    }

    BOTH Point operator+(const Vector& rhs) const {
        auto res = *this;
        return res += rhs;
    }

    BOTH Vector operator-(const Point& rhs) const {
        return mPos - rhs.mPos;
    }

    BOTH Point transform(const mat4& mat) const {
        return {
            mat[0][0] * mPos.x + mat[0][1] * mPos.y + mat[0][2] * mPos.z,
            mat[1][0] * mPos.x + mat[1][1] * mPos.y + mat[1][2] * mPos.z,
            mat[2][0] * mPos.x + mat[2][1] * mPos.y + mat[2][2] * mPos.z
        };
    }

    BOTH friend Point mix(const Point& lhs, const Point& rhs, const float w) {
        const auto vec = mix(lhs.mPos, rhs.mPos, w);
        return {vec.x, vec.y, vec.z};
    }
};

class Normal final {
private:
    Vector mNormal;
public:
    BOTH explicit Normal(const Vector& dir) : mNormal(normalize(dir)) {}

};

class Transform final {
private:
    mat4 mMat, mInv;
public:
    BOTH Transform(const mat4& mat, const mat4 inv) : mMat(mat), mInv(inv) {}
    BOTH explicit Transform(const mat4& mat) : mMat(mat), mInv(inverse(mat)) {}

    BOTH Transform& operator*=(const Transform& rhs) {
        mMat *= rhs.mMat;
        mInv *= rhs.mInv;
        return *this;
    }

    BOTH Transform operator*(const Transform& rhs) const {
        auto res = *this;
        return res *= rhs;
    }
};

struct Ray {
    vec3 origin;
    vec3 dir;
    float tMax;
    BOTH vec3 operator()(const float t) const {
        return origin * t;
    }
};

struct RayDifferential final : Ray {
    vec3 dodx, dddx, dody, dddy;
};

struct Bounds3 final {
    vec3 pMin, pMax;

    BOTH Bounds3() = default;
    BOTH Bounds3(const vec3 min, const vec3 max) : pMin(min), pMax(max) {}
    BOTH Bounds3 operator|(const Bounds3& rhs) const {
        return {min(pMin, rhs.pMin), max(pMax, rhs.pMax)};
    }

    BOTH Bounds3& operator|=(const Bounds3& rhs) {
        return *this = operator|(rhs);
    }

    BOTH Bounds3 operator&(const Bounds3& rhs) const {
        return {max(pMin, rhs.pMin), min(pMax, rhs.pMax)};
    }

    BOTH Bounds3& operator&=(const Bounds3& rhs) {
        return *this = operator&(rhs);
    }

    BOTH float area() const {
        const auto delta = pMax - pMin;
        return 2.0f * (delta.x * delta.y + delta.x * delta.z + delta.y * delta.z);
    }
};
