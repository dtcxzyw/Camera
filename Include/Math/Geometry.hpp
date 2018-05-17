#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <Math/EFloat.hpp>

BOTH Vector faceForward(const Vector& n,const Vector& v) {
    return dot(n, v) < 0.0f ? -n : n;
}

BOTH Vector halfVector(const Vector& in,const Vector& out) {
    return glm::normalize(in + out);
}

BOTH bool refract(const Vector& in,const Vector& normal,const float eta,Vector& out) {
    const auto cosThetaI = dot(in, normal);
    const auto sin2ThetaT = eta * eta*(1.0f - cosThetaI * cosThetaI);
    if (sin2ThetaT >= 1.0f)return false;
    const auto cosThetaT = sqrt(1.0f - sin2ThetaT);
    out = (eta*cosThetaI - cosThetaT) *normal - eta * in;
    return true;
}

BOTH int maxDim(const Vector& vec) {
    const auto vecAbs = abs(vec);
    return vecAbs.x > vecAbs.y ? (vecAbs.x > vecAbs.z ? 0 : 2) : (vecAbs.y > vecAbs.z ? 1 : 2);
}

BOTH Vector permute(const Vector& vec,const int x,const int y,const int z) {
    return { vec[x],vec[y],vec[z] };
}

struct Point final {
    union {
        Vector pos;

        struct {
            float x, y, z;
        };
    };

    BOTH Point() : pos(0.0f, 0.0f, 0.0f) {}
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

    BOTH Point operator+(const Point& rhs) const {
        return Point{pos + rhs.pos};
    }

    BOTH Point operator*(const float rhs) const {
        return Point{pos * rhs};
    }

    BOTH float operator[](const int id) const {
        return reinterpret_cast<const float*>(this)[id];
    }

    BOTH float& operator[](const int id) {
        return reinterpret_cast<float*>(this)[id];
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
    BOTH Vector operator*(const float rhs) const {
        return mNormal * rhs;
    }
    BOTH Normal operator-() const;
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

BOTH Normal reorthogonalize(const Normal& normal, const Normal& tangent) {
    const Vector n{normal};
    const Vector t{tangent};
    return Normal(t - dot(t, n) * n);
}

struct Ray final {
    Point origin;
    Vector dir;
    float tMax;
    Point xOri, yOri;
    Vector xDir, yDir;
    BOTH Ray(const Point& ori, const Vector& dir, const float tMax = std::numeric_limits<float>::max(),
        const Point& xOri = {}, const Vector& xDir = {}, const Point& yOri = {}, const Vector& yDir = {}) :
        origin(ori), dir(dir), tMax(tMax), xOri(xOri), xDir(xDir), yOri(yOri), yDir(yDir) {}

    BOTH Point operator()(const float t) const {
        return origin + dir * t;
    }
};

class Bounds final {
private:
    Point mMin;
    Point mMax;
public:
    BOTH Bounds() :mMin(Vector{ std::numeric_limits<float>::max() }), 
        mMax(Vector{ -std::numeric_limits<float>::max() }) {}
    BOTH explicit Bounds(const Point& pos) :mMin(pos), mMax(pos) {}
    BOTH Bounds(const Point& min, const Point& max) : mMin(min), mMax(max) {}
    BOTH Bounds operator|(const Bounds& rhs) const {
        return {min(mMin, rhs.mMin), max(mMax, rhs.mMax)};
    }

    BOTH Bounds& operator|=(const Bounds& rhs) {
        return *this = operator|(rhs);
    }

    BOTH Bounds operator&(const Bounds& rhs) const {
        return {max(mMin, rhs.mMin), min(mMax, rhs.mMax)};
    }

    BOTH Bounds& operator&=(const Bounds& rhs) {
        return *this = operator&(rhs);
    }

    BOTH float area() const {
        const auto delta = mMax - mMin;
        return 2.0f * (delta.x * delta.y + delta.x * delta.z + delta.y * delta.z);
    }

    BOTH Point operator[](const int id) const {
        return id ? mMax : mMin;
    }

    BOTH Point corner(const int id) const {
        return Point{operator[](id & 1).x, operator[](id & 2).y, operator[](id & 4).z};
    }

    DEVICE bool intersect(const Ray& ray, const float tHit, const Vector& invDir, 
        const glm::bvec3& neg) const {
        const auto& bounds = *this;
        const auto tMin = max3((bounds[neg.x].x - ray.origin.x) * invDir.x, 
            (bounds[neg.y].y - ray.origin.y) * invDir.y,
            (bounds[neg.z].z - ray.origin.z) * invDir.z);
        const auto tMax = min3((bounds[!neg.x].x - ray.origin.x) * invDir.x,
            (bounds[!neg.y].y - ray.origin.y) * invDir.y,
            (bounds[!neg.z].z - ray.origin.z) * invDir.z);
        return (tMin < tMax) & (tMin < tHit) & (tMax > 0.0f);
    }

    DEVICE bool empty() const {
        return (mMin.x > mMax.x) | (mMin.y > mMax.y) | (mMin.z > mMax.z);
    }
};

class Transform final {
private:
    glm::mat4 mMat, mInv;
public:
    BOTH Transform() = default;
    BOTH Transform(const glm::mat4& mat, const glm::mat4& inv) : mMat(mat), mInv(inv) {}
    BOTH explicit Transform(const glm::mat4& mat) : Transform(mat, inverse(mat)) {}

    BOTH friend Transform inverse(const Transform& transform) {
        return {transform.mInv, transform.mMat};
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

    BOTH Point operator()(const Point& rhs, Vector& err) const {
        const Vector pos(rhs);
        err = gamma(3) * Vector{
            fabs(mMat[0][0] * pos.x) + fabs(mMat[1][0] * pos.y) + fabs(mMat[2][0] * pos.z) + fabs(mMat[3][0]),
            fabs(mMat[0][1] * pos.x) + fabs(mMat[1][1] * pos.y) + fabs(mMat[2][1] * pos.z) + fabs(mMat[3][1]),
            fabs(mMat[0][2] * pos.x) + fabs(mMat[1][2] * pos.y) + fabs(mMat[2][2] * pos.z) + fabs(mMat[3][2])
        };
        return operator()(rhs);
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

    BOTH Ray operator()(const Ray& ray) const {
        Vector err;
        const auto ori = operator()(ray.origin, err);
        const auto dir = operator()(ray.dir);
        const auto dt = dot(abs(dir), err) / glm::length2(dir);
        return Ray(ori + dir * dt, dir, ray.tMax - dt,
            operator()(ray.xOri), operator()(ray.xDir), operator()(ray.yOri), operator()(ray.yDir));
    }

    BOTH Bounds operator()(const Bounds& bounds) const {
        const Bounds a(bounds.corner(0), bounds.corner(1));
        const Bounds b(bounds.corner(2), bounds.corner(3));
        const Bounds c(bounds.corner(4), bounds.corner(5));
        const Bounds d(bounds.corner(6), bounds.corner(7));
        return a | b | c | d;
    }
};

struct VertexDesc final {
    Point pos;
    Vector normal;
    Vector tangent;
    vec2 uv;
};
