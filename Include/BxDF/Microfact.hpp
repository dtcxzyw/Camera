#pragma once
#include <BxDF/BxDF.hpp>

template<typename T>
class MicrofactDistributionHelper {
private:
    CUDA const T& self() const {
        return *static_cast<const T*>(this);
    }
public:
    CUDA float calcG1(const Vector& w) const {
        return 1.0f / (1.0f + self().calcLambda(w));
    }
    CUDA float calcG(const Vector& wo,const Vector& wi) const {
        return 1.0f / (1.0f + self().calcLambda(wo) + self().calcLambda(wi));
    }
    CUDA float pdf(const Vector&wo, const Vector& wh) const {
        return self().calcD(wh)*calcG1(wo)*fabs(dot(wo, wh)) / absCosTheta(wo);
    }
};

class TrowbridgeReitzDistribution final :public MicrofactDistributionHelper<TrowbridgeReitzDistribution> {
private:
    float mAlphaX, mAlphaY;
    CUDA static float toAlpha(const float roughness) {
        const auto x = std::log(fmax(roughness, 1e-3f));
        return (((0.000640711f*x + 0.0171201f)*x + 0.1734f)*x + 0.819955f)*x + 1.62142f;
    }
    CUDA static vec2 sample11(const float cosTheta,const vec2 sample) {
        // special case (normal incidence)
        if (cosTheta > 0.9999f) {
            const auto r = sqrt(sample.x / (1.0f - sample.x));
            const auto phi = 6.28318530718 * sample.y;
            return { r*cos(phi),r*sin(phi) };
        }

        const auto sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        const auto tanTheta = sinTheta / cosTheta;
        const auto g1 = 2.0f / (1.0f + sqrt(1.0f + tanTheta * tanTheta));

        // sample slope.x
        const auto a = 2.0f * sample.x / g1 - 1.0f;
        const auto tmp = fmin(1e10f, 1.0f / (a * a - 1.0f));
        const auto b = tanTheta;
        const auto d = sqrt(fmax(b * b * tmp * tmp - (a * a - b * b) * tmp, 0.0f));
        const auto slopeX1 = b * tmp - d;
        const auto slopeX2 = b * tmp + d;
        const auto slopeX = (a < 0.0f | slopeX2 > 1.0f / tanTheta) ? slopeX1 : slopeX2;

        // sample slope.y
        const auto s = sample.y > 0.5f ? 1.0f : -1.0f;
        const auto u2 = 2.0f* fabs(sample.y - 0.5f);
        const auto z =
            (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) /
            (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        return { slopeX,s * z * sqrt(1.0f + slopeX * slopeX) };
    }
public:
    CUDA TrowbridgeReitzDistribution(const float rx, const float ry)
        :mAlphaX(toAlpha(rx)), mAlphaY(toAlpha(ry)) {}
    CUDA float calcD(const Vector& wh) const {
        const auto tan2ThetaH = tan2Theta(wh);
        if (std::isinf(tan2ThetaH)) return 0.0f;
        const auto cos4Theta = cos2Theta(wh) * cos2Theta(wh);
        const auto e =(cos2Phi(wh) / (mAlphaX * mAlphaX) + sin2Phi(wh) / (mAlphaY * mAlphaY)) *
            tan2ThetaH;
        return one_over_pi<float>() / (mAlphaX * mAlphaY * cos4Theta * (1 + e) * (1 + e));
    }
    CUDA float calcLambda(const Vector& w) const {
        const auto alpha2 = cosPhi(w)*mAlphaX*mAlphaX + sinPhi(w)*mAlphaY*mAlphaY;
        const auto tan2ThetaH = tan2Theta(w);
        return 0.5f*(-1.0f + sqrt(1.0f + alpha2 * tan2ThetaH));
    }
    CUDA Vector sampleWh(const Vector& wo,const vec2 sample) const {
        const auto filp = wo.z < 0.0f;
        const auto wi = filp ? -wo : wo;
        // 1. stretch wi
        const auto wiStretched = glm::normalize(Vector(mAlphaX * wi.x, mAlphaY * wi.y, wi.z));

        // 2. simulate P22_{wi}(slope.x, slope.y, 1, 1)
        auto slope=sample11(cosTheta(wiStretched), sample);

        // 3. rotate
        const auto tmp = cosPhi(wiStretched) * slope.x - sinPhi(wiStretched) * slope.y;
        slope.y = sinPhi(wiStretched) * slope.x + cosPhi(wiStretched) * slope.y;
        slope.x = tmp;

        // 4. unstretch
        slope.x *= mAlphaX;
        slope.y *= mAlphaY;

        // 5. compute normal
        const auto wh = glm::normalize(Vector(-slope.x, -slope.y, 1.0f));
        return filp ? -wh : wh;
    }
};

class MicrofactDistributionWarpper final {
private:
    union {
        TrowbridgeReitzDistribution tr;
    };
private:
    CUDA explicit MicrofactDistributionWarpper(const TrowbridgeReitzDistribution& tr) :tr(tr) {}
    CUDA float calcG(const Vector& wo, const Vector& wi) const {
        return tr.calcG(wo, wi);
    }
    CUDA float pdf(const Vector&wo, const Vector& wh) const {
        return tr.pdf(wo, wh);
    }
    CUDA float calcD(const Vector& h) const {
        return tr.calcD(h);
    }
    CUDA Vector sampleWh(const Vector& wo, const vec2 sample) const {
        return tr.sampleWh(wo, sample);
    }
};
