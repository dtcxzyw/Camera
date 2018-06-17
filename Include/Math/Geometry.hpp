#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <Math/EFloat.hpp>

inline BOTH Vector faceForward(const Vector& n, const Vector& v) {
    return dot(n, v) < 0.0f ? -n : n;
}

inline BOTH Vector halfVector(const Vector& in, const Vector& out) {
    return normalize(in + out);
}

inline BOTH bool refract(const Vector& in, const Vector& normal, const float eta, Vector& out) {
    const auto cosThetaI = dot(in, normal);
    const auto sin2ThetaT = eta * eta * fmax(0.0f, 1.0f - cosThetaI * cosThetaI);
    if (sin2ThetaT >= 1.0f)return false;
    const auto cosThetaT = sqrt(1.0f - sin2ThetaT);
    out = (eta * cosThetaI - cosThetaT) * normal - eta * in;
    return true;
}

inline BOTH int maxDim(const Vector& vec) {
    const auto vecAbs = abs(vec);
    return vecAbs.x > vecAbs.y ? (vecAbs.x > vecAbs.z ? 0 : 2) : (vecAbs.y > vecAbs.z ? 1 : 2);
}

inline BOTH Vector permute(const Vector& vec, const int x, const int y, const int z) {
    return {vec[x], vec[y], vec[z]};
}

struct Point final {
    union {
        Vector pos;

        struct {
            float x, y, z;
        };
    };

    BOTH    Point() : pos(0.0f, 0.0f, 0.0f) {}
    BOTH    explicit Point(const Vector& pos) : pos(pos) {}
    BOTH    Point(const float x, const float y, const float z) : pos(x, y, z) {}

    BOTH    Point(const Point& rhs) : pos(rhs.pos) {}
    BOTH    Point& operator=(const Point& rhs) {
        pos = rhs.pos;
        return *this;
    }

    BOTH    Point& operator+=(const Vector& rhs) {
        pos += rhs;
        return *this;
    }

    BOTH    Point operator+(const Vector& rhs) const {
        auto res = *this;
        return res += rhs;
    }

    BOTH    Point& operator-=(const Vector& rhs) {
        pos -= rhs;
        return *this;
    }

    BOTH    Point operator-(const Vector& rhs) const {
        auto res = *this;
        return res -= rhs;
    }

    BOTH    Vector operator-(const Point& rhs) const {
        return pos - rhs.pos;
    }

    BOTH    bool operator==(const Point& rhs) const {
        return pos == rhs.pos;
    }

    BOTH    bool operator!=(const Point& rhs) const {
        return pos != rhs.pos;
    }

    BOTH    explicit operator Vector() const {
        return pos;
    }

    BOTH    friend Point mix(const Point& lhs, const Point& rhs, const float w) {
        return Point{mix(lhs.pos, rhs.pos, w)};
    }

    BOTH    friend Point mix(const Point& lhs, const Point& rhs, const Vector& w) {
        return Point{mix(lhs.pos, rhs.pos, w)};
    }

    BOTH    friend Point min(const Point& lhs, const Point& rhs) {
        return Point{min(lhs.pos, rhs.pos)};
    }

    BOTH    friend Point max(const Point& lhs, const Point& rhs) {
        return Point{max(lhs.pos, rhs.pos)};
    }

    BOTH    friend float distance(const Point& lhs, const Point& rhs) {
        return glm::distance(lhs.pos, rhs.pos);
    }

    BOTH    friend float distance2(const Point& lhs, const Point& rhs) {
        return glm::distance2(lhs.pos, rhs.pos);
    }

    BOTH    Point operator+(const Point& rhs) const {
        return Point{pos + rhs.pos};
    }

    BOTH    Point operator*(const float rhs) const {
        return Point{pos * rhs};
    }

    BOTH    float operator[](const int id) const {
        return reinterpret_cast<const float*>(this)[id];
    }

    BOTH    float& operator[](const int id) {
        return reinterpret_cast<float*>(this)[id];
    }
};

class Normal final {
private:
    Vector mNormal;
public:
    Normal() = default;
    BOTH    explicit Normal(const Vector& dir) : mNormal(dir) {}
    BOTH    explicit operator Vector() const {
        return mNormal;
    }

    BOTH    Normal operator*(const float rhs) const {
        return Normal(mNormal * rhs);
    }

    BOTH    Normal operator-() const {
        return Normal{-mNormal};
    }

    BOTH    float operator[](const int i) const {
        return mNormal[i];
    }

    BOTH    Normal operator+(const Normal& rhs) const {
        return Normal{mNormal + rhs.mNormal};
    }

    BOTH    Normal operator-(const Normal& rhs) const {
        return Normal{mNormal - rhs.mNormal};
    }
};

inline BOTH Normal normalize(const Normal& a) {
    return Normal{normalize(Vector{a})};
}

inline BOTH Normal halfVector(const Normal& a, const Normal& b) {
    return Normal{normalize(a) + normalize(b)};
}

inline BOTH float dot(const Normal& a, const Normal& b) {
    return dot(Vector{a}, Vector{b});
}

inline BOTH Normal reflect(const Normal& a, const Normal& b) {
    return Normal{glm::reflect(Vector{a}, Vector{b})};
}

inline BOTH Normal cross(const Normal& a, const Normal& b) {
    return Normal{cross(Vector{a}, Vector{b})};
}

inline BOTH Normal faceForward(const Normal& n, const Normal& v) {
    return dot(n, v) < 0.0f ? -n : n;
}

struct Ray {
    Point origin;
    Vector dir;
    float tMax;
    BOTH    Ray(const Point& ori, const Vector& dir, const float tMax = std::numeric_limits<float>::max()) :
        origin(ori), dir(dir), tMax(tMax) {}

    BOTH    Point operator()(const float t) const {
        return origin + dir * t;
    }
};

struct RayDifferential final : public Ray {
    Point xOri, yOri;
    Vector xDir, yDir;
    bool hasDifferentials;

    BOTH    RayDifferential(const Point& ori, const Vector& dir,
        const float tMax = std::numeric_limits<float>::max()) :
        Ray(ori, dir, tMax), xOri(xOri), yOri(yOri), xDir(xDir), yDir(yDir), hasDifferentials(false) {}

    BOTH explicit RayDifferential(const Ray& ray) : Ray(ray), hasDifferentials(false) {}
};

class Bounds final {
private:
    Point mMin;
    Point mMax;
public:
    BOTH    Bounds() : mMin(Vector{std::numeric_limits<float>::max()}),
        mMax(Vector{-std::numeric_limits<float>::max()}) {}

    BOTH    explicit Bounds(const Point& pos) : mMin(pos), mMax(pos) {}
    BOTH    Bounds(const Point& a, const Point& b) : mMin(min(a, b)), mMax(max(a, b)) {}
    BOTH    Bounds operator|(const Bounds& rhs) const {
        return {min(mMin, rhs.mMin), max(mMax, rhs.mMax)};
    }

    BOTH    Bounds& operator|=(const Bounds& rhs) {
        return *this = operator|(rhs);
    }

    BOTH    Bounds operator&(const Bounds& rhs) const {
        return {max(mMin, rhs.mMin), min(mMax, rhs.mMax)};
    }

    BOTH    Bounds& operator&=(const Bounds& rhs) {
        return *this = operator&(rhs);
    }

    BOTH    Point corner(const int id) const {
        return Point{operator[](id & 1).x, operator[](id & 2).y, operator[](id & 4).z};
    }

    BOTH    float area() const {
        const auto delta = mMax - mMin;
        return 2.0f * (delta.x * delta.y + delta.x * delta.z + delta.y * delta.z);
    }

    BOTH    Point operator[](const int id) const {
        return id ? mMax : mMin;
    }

    BOTH    Point lerp(const Vector& w) const {
        return mix(mMin, mMax, w);
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
        return (tMin <= tMax) & (tMin < tHit) & (tMax > 0.0f);
    }

    std::pair<Point, float> boundingSphere() const {
        return std::make_pair(mix(mMin, mMax, 0.5f), distance(mMin, mMax)*0.5f);
    }
};

class Transform final {
private:
    glm::mat4 mMat, mInv;
public:
    Transform() = default;
    BOTH    Transform(const glm::mat4& mat, const glm::mat4& inv) : mMat(mat), mInv(inv) {}
    BOTH    explicit Transform(const glm::mat4& mat) : Transform(mat, inverse(mat)) {}

    BOTH    friend Transform inverse(const Transform& transform) {
        return {transform.mInv, transform.mMat};
    }

    BOTH    Transform& operator*=(const Transform& rhs) {
        mMat *= rhs.mMat;
        mInv = rhs.mInv * mInv;
        return *this;
    }

    BOTH    Transform operator*(const Transform& rhs) const {
        auto res = *this;
        return res *= rhs;
    }

    BOTH    Vector operator()(const Vector& rhs) const {
        return {
            mMat[0][0] * rhs.x + mMat[1][0] * rhs.y + mMat[2][0] * rhs.z,
            mMat[0][1] * rhs.x + mMat[1][1] * rhs.y + mMat[2][1] * rhs.z,
            mMat[0][2] * rhs.x + mMat[1][2] * rhs.y + mMat[2][2] * rhs.z
        };
    }

    BOTH    Vector operator()(const Vector& v, Vector& absError) const {
        absError.x = gamma(3) * (fabs(mMat[0][0] * v.x) + fabs(mMat[0][1] * v.y) + fabs(mMat[0][2] * v.z));
        absError.y = gamma(3) * (fabs(mMat[1][0] * v.x) + fabs(mMat[1][1] * v.y) + fabs(mMat[1][2] * v.z));
        absError.z = gamma(3) * (fabs(mMat[2][0] * v.x) + fabs(mMat[2][1] * v.y) + fabs(mMat[2][2] * v.z));
        return (*this)(v);
    }

    BOTH    Point operator()(const Point& rhs) const {
        return {
            mMat[0][0] * rhs.x + mMat[1][0] * rhs.y + mMat[2][0] * rhs.z + mMat[3][0],
            mMat[0][1] * rhs.x + mMat[1][1] * rhs.y + mMat[2][1] * rhs.z + mMat[3][1],
            mMat[0][2] * rhs.x + mMat[1][2] * rhs.y + mMat[2][2] * rhs.z + mMat[3][2]
        };
    }

    BOTH    Point operator()(const Point& rhs, Vector& err) const {
        err = gamma(3) * Vector{
            fabs(mMat[0][0] * rhs.x) + fabs(mMat[1][0] * rhs.y) + fabs(mMat[2][0] * rhs.z) + fabs(mMat[3][0]),
            fabs(mMat[0][1] * rhs.x) + fabs(mMat[1][1] * rhs.y) + fabs(mMat[2][1] * rhs.z) + fabs(mMat[3][1]),
            fabs(mMat[0][2] * rhs.x) + fabs(mMat[1][2] * rhs.y) + fabs(mMat[2][2] * rhs.z) + fabs(mMat[3][2])
        };
        return operator()(rhs);
    }

    BOTH    Point operator()(const Point& rhs, const Vector& inErr, Vector& outErr) const {
        const auto res = operator()(rhs, outErr);
        outErr += (gamma(3) + 1.0f) * Vector{
            fabs(mMat[0][0] * inErr.x) + fabs(mMat[0][1] * inErr.y) + fabs(mMat[0][2] * inErr.z),
            fabs(mMat[1][0] * inErr.x) + fabs(mMat[1][1] * inErr.y) + fabs(mMat[1][2] * inErr.z),
            fabs(mMat[2][0] * inErr.x) + fabs(mMat[2][1] * inErr.y) + fabs(mMat[2][2] * inErr.z)
        };
        return res;
    }

    //mat'=mat3(transpose(inverse(mat)))
    BOTH    Normal operator()(const Normal& rhs) const {
        const Vector normal(rhs);
        return Normal{
            Vector{
                mInv[0][0] * normal.x + mInv[0][1] * normal.y + mInv[0][2] * normal.z,
                mInv[1][0] * normal.x + mInv[1][1] * normal.y + mInv[1][2] * normal.z,
                mInv[2][0] * normal.x + mInv[2][1] * normal.y + mInv[2][2] * normal.z
            }
        };
    }

    BOTH    Ray operator()(const Ray& ray, Vector& oErr, Vector& dErr) const {
        const auto ori = (*this)(ray.origin, oErr);
        const auto dir = (*this)(ray.dir, dErr);
        const auto dt = dot(Vector{abs(dir)}, oErr) / glm::length2(dir);
        return Ray{ori + dir * dt, dir, ray.tMax - dt};
    }

    BOTH    Ray operator()(const Ray& ray) const {
        Vector oErr;
        const auto ori = (*this)(ray.origin, oErr);
        const auto dir = (*this)(ray.dir);
        const auto dt = dot(Vector{abs(dir)}, oErr) / glm::length2(dir);
        return Ray{ ori + dir * dt, dir, ray.tMax - dt };
    }

    BOTH    RayDifferential operator()(const RayDifferential& ray) const {
        RayDifferential res{ (*this)(static_cast<const Ray&>(ray)) };
        if (ray.hasDifferentials) {
            res.xOri = (*this)(ray.xOri); 
            res.xDir = (*this)(ray.xDir);
            res.yOri = (*this)(ray.yOri);
            res.yDir = (*this)(ray.yDir);
            res.hasDifferentials = true;
        }
        return res;
    }

    BOTH    Bounds operator()(const Bounds& bounds) const {
        auto&& trans = *this;
        const Bounds a(trans(bounds.corner(0)), trans(bounds.corner(1)));
        const Bounds b(trans(bounds.corner(2)), trans(bounds.corner(3)));
        const Bounds c(trans(bounds.corner(4)), trans(bounds.corner(5)));
        const Bounds d(trans(bounds.corner(6)), trans(bounds.corner(7)));
        return a | b | c | d;
    }
};

struct VertexDesc final {
    Point pos;
    Normal normal;
    Normal tangent;
    vec2 uv;
};

inline BOTH float sphericalTheta(const Vector& v) {
    return acos(clamp(v.z, -1.0f, 1.0f));
}

inline BOTH float sphericalPhi(const Vector& v) {
    const auto p = std::atan2(v.y, v.x);
    return (p < 0.0f) ? (p + two_pi<float>()) : p;
}

inline DEVICE void defaultCoordinateSystem(const Vector& n, Vector& t, Vector& b) {
    if (fabs(n.x) > fabs(n.y))
        t = Vector(-n.z, 0.0f, n.x) / std::sqrt(n.x * n.x + n.z * n.z);
    else
        t = Vector(0.0f, n.z, -n.y) / std::sqrt(n.y * n.y + n.z * n.z);
    b = cross(n, t);
}

inline DEVICE Vector sphericalDirection(const float sinTheta, const float cosTheta, const float phi,
    const Vector& x, const Vector& y, const Vector& z) {
    return sinTheta * cos(phi) * x + sinTheta * sin(phi) * y + cosTheta * z;
}
