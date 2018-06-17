#pragma once
#include <Light/Shapes/ShapeShared.hpp>
#include <Sampler/Sampling.hpp>

class Sphere final :public ShapeHelper<Sphere> {
private:
    float mRadius, mZMin, mZMax, mThetaMin, mThetaMax, mPhiMax, mRadius2, mInvArea;
    Point mCenter;
    DEVICE bool test(const Ray& r, float& tHit, Point& pHit) const {
        Vector oErr, dErr;
        const auto ray = inverse(toWorld)(r, oErr, dErr);

        const auto ox = makeEFloat(ray.origin.x, oErr.x);
        const auto oy = makeEFloat(ray.origin.y, oErr.y);
        const auto oz = makeEFloat(ray.origin.z, oErr.z);
        const auto dx = makeEFloat(ray.dir.x, dErr.x);
        const auto dy = makeEFloat(ray.dir.y, dErr.y);
        const auto dz = makeEFloat(ray.dir.z, dErr.z);
        const auto a = dx * dx + dy * dy + dz * dz;
        const auto b = (dx * ox + dy * oy + dz * oz)*EFloat { 2.0f };
        const auto c = ox * ox + oy * oy + oz * oz - EFloat{ mRadius2 };

        EFloat t0, t1;
        if (!quadratic(a, b, c, t0, t1)) return false;

        const auto calcHit = [&](const EFloat& t)->bool {
            if (t.lowerBound() <= 0.0f | t.upperBound() > tHit)return false;
            pHit = ray(t.lowerBound());
            pHit = pHit * (mRadius / length(Vector{ pHit }));

            if (pHit.x == 0 & pHit.y == 0) pHit.x = 1e-5f * mRadius;
            auto phi = std::atan2(pHit.y, pHit.x);
            phi = phi < 0.0f ? phi + two_pi<float>() : phi;
            if(mZMin <= pHit.z & pHit.z <= mZMax & phi <= mPhiMax) {
                tHit = t.lowerBound();
                return true;
            }
            return false;
        };
        if (!calcHit(t0))return calcHit(t1);
        return true;
    }
public:
    Sphere(const Transform& trans, const float radius, const float zMin = -std::numeric_limits<float>::max(),
        const float zMax = std::numeric_limits<float>::max(),
        const float phiMax = 360.0f, const bool reverseOri = false)
        : ShapeHelper(trans, reverseOri), mRadius(radius),
        mZMin(clamp(fmin(zMin, zMax), -radius, radius)),
        mZMax(clamp(fmax(zMin, zMax), -radius, radius)),
        mThetaMin(acos(clamp(fmin(zMin, zMax) / radius, -1.0f, 1.0f))),
        mThetaMax(acos(clamp(fmax(zMin, zMax) / radius, -1.0f, 1.0f))),
        mPhiMax(glm::radians(clamp(phiMax, 0.0f, 360.0f))), mRadius2(radius * radius),
        mCenter(trans(Point{})), mInvArea(1.0f / (mPhiMax * mRadius * (mZMax - mZMin))) {}

    DEVICE Interaction sample(const vec2 sample, float& pdf) const {
        const Point pObj{mRadius * uniformSampleSphere(sample)};
        Interaction it;
        it.localGeometry.normal = normalize(toWorld(Normal{Vector{pObj}}));
        if (reverseOri) it.localGeometry.normal = -it.localGeometry.normal;
        it.pos = toWorld(pObj);
        pdf = mInvArea;
        return it;
    }

    DEVICE Interaction sample(const Interaction& isect, const vec2 sample, float& pdf) const {
        const auto pOrigin = isect.calcOffsetOrigin(mCenter - isect.pos);
        if (distance2(pOrigin, mCenter) <= mRadius2) {
            const auto intr = Sphere::sample(sample, pdf);
            const auto wi = intr.pos - isect.pos;
            if (glm::length2(wi) == 0.0f)pdf = 0.0f;
            else {
                pdf *= distance2(isect.pos, intr.pos) /
                    fabs(dot(Vector{intr.localGeometry.normal }, -normalize(wi)));
            }
            if (isinf(pdf)) pdf = 0.0f;
            return intr;
        }

        const auto wc = normalize(mCenter - isect.pos);
        Vector wcX, wcY;
        defaultCoordinateSystem(wc, wcX, wcY);

        const auto sinThetaMax2 = mRadius2 / distance2(isect.pos, mCenter);
        const auto cosThetaMax = sqrt(fmax(0.0f, 1.0f - sinThetaMax2));
        const auto cosTheta = 1.0f - sample.x + sample.x * cosThetaMax;
        const auto sinTheta = sqrt(fmax(0.0f, 1.0f - cosTheta * cosTheta));
        const auto phi = sample.y * two_pi<float>();

        const auto dc = distance(isect.pos, mCenter);
        const auto ds = dc * cosTheta - sqrt(fmax(0.0f, mRadius2 - dc * dc * sinTheta * sinTheta));
        const auto cosAlpha = (dc * dc + mRadius2 - ds * ds) / (2 * dc * mRadius);
        const auto sinAlpha = sqrt(fmax(0.0f, 1.0f - cosAlpha * cosAlpha));

        const auto nWorld = sphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
        const auto pWorld = mCenter + Point{nWorld * mRadius};

        Interaction it;
        it.pos = pWorld;
        it.localGeometry.normal = Normal{nWorld};
        if (reverseOri) it.localGeometry.normal = -it.localGeometry.normal;

        pdf = uniformConePdf(cosThetaMax);
        return it;
    }

    DEVICE float pdf(const Interaction& isect, const Vector& wi) const {
        const auto pOrigin = isect.calcOffsetOrigin(mCenter - isect.pos);
        if (distance2(pOrigin, mCenter) <= mRadius2)
            return ShapeHelper::pdf(isect, wi);

        const auto sinThetaMax2 = mRadius2 / distance2(isect.pos, mCenter);
        const auto cosThetaMax = sqrt(fmax(0.0f, 1.0f - sinThetaMax2));
        return uniformConePdf(cosThetaMax);
    }

    DEVICE bool intersect(const Ray& r) const {
        Point pHit;
        auto tHit = r.tMax;
        return test(r, tHit, pHit);
    }

    DEVICE bool intersect(const Ray& r, float& tHit, Interaction& isect) const {
        Point pHit;
        if (!test(r, tHit, pHit))return false;

        const auto theta = acos(clamp(pHit.z / mRadius, -1.0f, 1.0f));

        const auto zRadius = sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
        const auto invZRadius = 1.0f / zRadius;
        const auto cosPhi = pHit.x * invZRadius;
        const auto sinPhi = pHit.y * invZRadius;
        const Vector dpdu(-mPhiMax * pHit.y, mPhiMax * pHit.x, 0.0f);
        const auto dpdv = (mThetaMax - mThetaMin) *
            Vector {pHit.z * cosPhi, pHit.z * sinPhi, -mRadius * sin(theta)};
        isect.pos = pHit;
        isect.localGeometry.normal = Normal{ normalize(cross(dpdu, dpdv)) };
        return true;
    }

    DEVICE float invArea() const {
        return mInvArea;
    }
};
