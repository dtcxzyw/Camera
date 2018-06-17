#pragma once
#include <Spectrum/SpectrumConfig.hpp>

class FresnelDielectric final {
private:
    float mEtaI, mEtaT;
public:
    DEVICE FresnelDielectric(const float etaI, const float etaT) : mEtaI(etaI), mEtaT(etaT) {}
    DEVICE float f(float cosThetaI) const {
        cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
        auto etaI = mEtaI, etaT = mEtaT;
        if (cosThetaI <= 0.0f) {
            cosThetaI = -cosThetaI;
            cudaSwap(etaI, etaT);
        }
        const auto sinThetaT = etaI / etaT * sqrt(fmax(0.0f, 1.0f - cosThetaI * cosThetaI));

        if (sinThetaT >= 1.0f)return 1.0f;

        const auto cosThetaT = sqrt(fmax(0.0f, 1.0f - sinThetaT * sinThetaT));
        const auto ii = etaI * cosThetaI, it = etaI * cosThetaT,
            ti = etaT * cosThetaI, tt = etaT * cosThetaT;
        const auto a = (ti - it) / (ti + it);
        const auto b = (ii - tt) / (ii + tt);
        const auto res = 0.5f * (a * a + b * b);
        CHECKFP(res);
        return res;
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

        const auto res = 0.5f * (rp + rs);
        CHECKFP(res.y());
        return res;
    }
};

class FresnelNoOp final {};

class FresnelWrapper final {
private:
    union {
        FresnelNoOp noOp;
        FresnelDielectric dielectric;
        FresnelConductor conductor;
    };

    bool mIsDielectric, mIsConstant;
public:
    DEVICE explicit FresnelWrapper(const FresnelDielectric& dielectric)
        : dielectric(dielectric), mIsDielectric(true), mIsConstant(false) {}

    DEVICE explicit FresnelWrapper(const FresnelConductor& conductor)
        : conductor(conductor), mIsDielectric(false), mIsConstant(false) {}

    DEVICE explicit FresnelWrapper(const FresnelNoOp& fresnel)
        : noOp(fresnel), mIsDielectric(true), mIsConstant(true) {}

    DEVICE FresnelWrapper(const FresnelWrapper& rhs) {
        memcpy(this, &rhs, sizeof(FresnelWrapper));
    }

    DEVICE Spectrum f(const float cosThetaI) const {
        if (mIsConstant)return Spectrum{1.0f};
        return mIsDielectric ? Spectrum{dielectric.f(cosThetaI)} : conductor.f(cosThetaI);
    }
};
