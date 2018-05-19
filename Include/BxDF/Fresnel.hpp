#pragma once
#include <Math/Geometry.hpp>
#include <Spectrum/SpectrumConfig.hpp>

class FresnelDielectric final {
private:
    float mEtaI, mEtaT;
public:
    DEVICE FresnelDielectric(const float etaI, const float etaT) : mEtaI(etaI), mEtaT(etaT) {}
    DEVICE float f(float cosThetaI) const {
        const auto etaI = mEtaI, etaT = mEtaT;
        if (cosThetaI < 0.0f) {
            cosThetaI = -cosThetaI;
            cudaSwap(etaI, etaT);
        }
        const auto sinThetaT = fmin(1.0f, etaI / etaT * sqrt(1.0f - cosThetaI * cosThetaI));
        const auto cosThetaT = sqrt(1.0f - sinThetaT * sinThetaT);
        const auto ii = cosThetaI * etaI, it = cosThetaI * etaT,
            ti = cosThetaT * etaI, tt = cosThetaT * etaT;
        const auto a = (ti - it) / (ti + it);
        const auto b = (ii - tt) / (ii + tt);
        return 0.5f * (a * a + b * b);
    }
};

//https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
class FresnelConductor final {
private:
    Spectrum mEtaI, mEtaT, mK;
public:
    DEVICE FresnelConductor(const Spectrum& etaI,
        const Spectrum& etaT, const Spectrum& k) : mEtaI(etaI), mEtaT(etaT), mK(k) {}

    DEVICE Spectrum f(const float cosThetaI) const {
        const auto eta = mEtaT / mEtaI;
        const auto etak = mK / mEtaI;

        const auto cosThetaI2 = cosThetaI * cosThetaI;
        const auto sinThetaI2 = 1.0f - cosThetaI2;
        const auto eta2 = eta * eta;
        const auto etak2 = etak * etak;

        const auto t0 = eta2 - etak2 - sinThetaI2;
        const auto a2Plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
        const auto t1 = a2Plusb2 + cosThetaI2;
        const auto a = sqrt(0.5f * (a2Plusb2 + t0));
        const auto t2 = 2.0f * cosThetaI * a;
        const auto rs = (t1 - t2) / (t1 + t2);

        const auto t3 = cosThetaI2 * a2Plusb2 + sinThetaI2 * sinThetaI2;
        const auto t4 = t2 * sinThetaI2;
        const auto rp = rs * (t3 - t4) / (t3 + t4);

        return 0.5f * (rp + rs);
    }
};

class FresnelWrapper final {
private:
    union {
        FresnelDielectric dielectric;
        FresnelConductor conductor;
    };

    bool mIsDielectric;
public:
    DEVICE explicit FresnelWrapper(const FresnelDielectric& dielectric)
        : dielectric(dielectric), mIsDielectric(true) {}

    DEVICE explicit FresnelWrapper(const FresnelConductor& conductor)
        : conductor(conductor), mIsDielectric(true) {}

    DEVICE Spectrum f(const float cosThetaI) const {
        return mIsDielectric ? Spectrum(dielectric.f(cosThetaI)) : conductor.f(cosThetaI);
    }
};
