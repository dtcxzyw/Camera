#pragma once
#include <Base/Common.hpp>

constexpr auto invpi = one_over_pi<float>();

inline CUDA glm::vec3 calcHalf(glm::vec3 in, glm::vec3 out) {
    return glm::normalize(in + out);
}

inline CUDA float fresnelSchlick(float f0, float idh) {
    idh = 1.0f - idh;
    float idh2 = idh*idh;
    return f0 + (1.0f - f0)*idh*idh2*idh2;
}

inline CUDA float GGXD(float ndh, float a) {
    float snh = sin(acos(ndh));
    float root = a / (a*a*ndh*ndh + snh*snh);
    return invpi*root*root;
}

inline CUDA float GGXG(float ndv, float roughness) {
    auto r = roughness + 1.0f;
    auto k = r*r / 8.0f;
    auto div = ndv * (1.0f - k) + k;
    return ndv / div;
}

inline CUDA float smithG(float ndi, float ndo, float roughness) {
    return GGXG(ndi, roughness) * GGXG(ndo, roughness);
}

//https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
inline CUDA float diffuse(float ndi, float ndo, float idh, float roughness) {
    auto cosi = 1.0f - fabs(ndi);
    auto cosi2 = cosi*cosi;
    auto coso = 1.0f - fabs(ndo);
    auto coso2 = coso*coso;
    auto fd90sub1 = 2.0f*idh*idh*roughness - 1.0f;
    return invpi*(1.0f + fd90sub1*cosi*cosi2*cosi2)
        *(1.0f + fd90sub1*coso*coso2*coso2);
}

inline CUDA float cookTorrance(float diff, float D, float F, float G, float ndi, float ndo) {
    return diff + D*F*G / (4.0f*fmax(ndi,0.0f)*fmax(ndo,0.0f)+epsilon<float>());
}

inline CUDA float calcWeight(float diff, float D, float F, float G, float ndi, float ndo,float dis2) {
    return pi<float>()*cookTorrance(diff, D, F, G, ndi, ndo)*ndi / dis2;
}
