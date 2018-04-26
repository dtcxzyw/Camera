#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

namespace Impl{
    template<typename First, typename... Others>
    class DataSet;
}

class Point final {
public:
    union {
        Vector pos;

        struct {
            float x, y, z;
        };
    };

    BOTH Point() :pos(0.0f, 0.0f, 0.0f) {}
    BOTH explicit Point(const Vector& pos) : pos(pos) {}
    BOTH Point(const float x, const float y, const float z) : pos(x, y, z) {}

    BOTH Point(const Point& rhs) : pos(rhs.pos) {}
    BOTH Point& operator=(const Point& rhs) {
        pos = rhs.pos;
        return *this;
    }

    BOTH Point& operator+=(const Vector& rhs) {
        pos += rhs;
        return *this;
    }

    BOTH Point operator+(const Vector& rhs) const {
        auto res = *this;
        return res += rhs;
    }

    BOTH Vector operator-(const Point& rhs) const {
        return pos - rhs.pos;
    }

    BOTH bool operator==(const Point& rhs) const {
        return pos == rhs.pos;
    }

    BOTH bool operator!=(const Point& rhs) const {
        return pos != rhs.pos;
    }

    BOTH explicit operator Vector() const {
        return pos;
    }

    BOTH friend Point mix(const Point& lhs, const Point& rhs, const float w) {
        return Point{mix(lhs.pos, rhs.pos, w)};
    }

    BOTH friend Point min(const Point& lhs, const Point& rhs) {
        return Point{min(lhs.pos, rhs.pos)};
    }

    BOTH friend Point max(const Point& lhs, const Point& rhs) {
        return Point{max(lhs.pos, rhs.pos)};
    }
private:
    template<typename First, typename... Others>
    friend class Impl::DataSet;
    BOTH Point operator+(const Point& rhs) const {
        return Point{ pos + rhs.pos };
    }
    BOTH Point operator*(const float rhs) const {
        return Point{ pos*rhs };
    }
};

class Normal final {
private:
    Vector mNormal;
public:
    BOTH Normal() = default;
    BOTH explicit Normal(const Vector& dir) : mNormal(normalize(dir)) {}
    BOTH explicit operator Vector() const {
        return mNormal;
    }
    BOTH Normal Normal::operator-() const;
};

BOTH Normal makeNormalUnsafe(const Vector& dir) {
    return *reinterpret_cast<const Normal*>(&dir);
}

BOTH Normal Normal::operator-() const {
    return makeNormalUnsafe(-mNormal);
}

BOTH Normal halfVector(const Normal& a, const Normal& b) {
    return Normal{Vector{a} + Vector{b}};
}

BOTH float dot(const Normal& a, const Normal& b) {
    return dot(Vector{a}, Vector{b});
}

BOTH Normal cross(const Normal& a, const Normal& b) {
    return Normal{cross(Vector{a}, Vector{b})};
}

BOTH Normal crossUnsafe(const Normal& a, const Normal& b) {
    return makeNormalUnsafe(cross(Vector{a}, Vector{b}));
}

BOTH Normal reflect(const Normal& in, const Normal& normal) {
    return makeNormalUnsafe(glm::reflect(Vector(in), Vector(normal)));
}

BOTH Normal reorthogonalize(const Normal& normal,const Normal& tangent) {
    const Vector n{ normal };
    const Vector t{ tangent };
    return Normal(t - dot(t, n) * n);
}

class Ray final {
private:
    Point mOrigin;
    Vector mDir;
    float mMaxT;
    Vector mDodx, mDddx, mDody, mDddy;
public:
    BOTH Ray(const Point& ori, const Vector& dir, const float tMax = std::numeric_limits<float>::max(),
        const Vector& dodx = {}, const Vector& dddx = {}, const Vector& dody = {}, const Vector& dddy = {}) :
        mOrigin(ori), mDir(dir), mMaxT(tMax),
        mDodx(dodx), mDddx(dddx), mDody(dody), mDddy(dddy) {}

    BOTH float t() const {
        return mMaxT;
    }

    BOTH void updateMaxT(const float t) {
        mMaxT = t;
    }

    BOTH Point operator()(const float t) const {
        return mOrigin + mDir * t;
    }
};

class Bounds3 final {
private:
    Point mMin, mMax;
public:
    BOTH Bounds3() = default;
    BOTH Bounds3(const Point& min, const Point& max) : mMin(min), mMax(max) {}
    BOTH Bounds3 operator|(const Bounds3& rhs) const {
        return {min(mMin, rhs.mMin), max(mMax, rhs.mMax)};
    }

    BOTH Bounds3& operator|=(const Bounds3& rhs) {
        return *this = operator|(rhs);
    }

    BOTH Bounds3 operator&(const Bounds3& rhs) const {
        return {max(mMin, rhs.mMin), min(mMax, rhs.mMax)};
    }

    BOTH Bounds3& operator&=(const Bounds3& rhs) {
        return *this = operator&(rhs);
    }

    BOTH float area() const {
        const auto delta = mMax - mMin;
        return 2.0f * (delta.x * delta.y + delta.x * delta.z + delta.y * delta.z);
    }
};

class Transform final {
private:
    glm::mat4 mMat, mInv;
public:
    BOTH Transform() = default;
    BOTH Transform(const glm::mat4& mat, const glm::mat4& inv) : mMat(mat), mInv(inv) {}
    BOTH explicit Transform(const glm::mat4& mat) : Transform(mat,inverse(mat)) {}

    BOTH friend Transform inverse(const Transform& transform) {
        return { transform.mInv, transform.mMat };
    }

    BOTH Transform& operator*=(const Transform& rhs) {
        mMat *= rhs.mMat;
        mInv *= rhs.mInv;
        return *this;
    }

    BOTH Transform operator*(const Transform& rhs) const {
        auto res = *this;
        return res *= rhs;
    }

    BOTH Vector operator()(const Vector& rhs) const {
        return {
            mMat[0][0] * rhs.x + mMat[1][0] * rhs.y + mMat[2][0] * rhs.z,
            mMat[0][1] * rhs.x + mMat[1][1] * rhs.y + mMat[2][1] * rhs.z,
            mMat[0][2] * rhs.x + mMat[1][2] * rhs.y + mMat[2][2] * rhs.z
        };
    }

    BOTH Point operator()(const Point& rhs) const {
        const Vector pos(rhs);
        return {
            mMat[0][0] * pos.x + mMat[1][0] * pos.y + mMat[2][0] * pos.z + mMat[3][0],
            mMat[0][1] * pos.x + mMat[1][1] * pos.y + mMat[2][1] * pos.z + mMat[3][1],
            mMat[0][2] * pos.x + mMat[1][2] * pos.y + mMat[2][2] * pos.z + mMat[3][2]
        };
    }

    //mat'=mat3(transpose(inverse(mat)))
    BOTH Normal operator()(const Normal& rhs) const {
        const Vector normal(rhs);
        return makeNormalUnsafe(
            {
                mInv[0][0] * normal.x + mInv[0][1] * normal.y + mInv[0][2] * normal.z,
                mInv[1][0] * normal.x + mInv[1][1] * normal.y + mInv[1][2] * normal.z,
                mInv[2][0] * normal.x + mInv[2][1] * normal.y + mInv[2][2] * normal.z
            });
    }
};
