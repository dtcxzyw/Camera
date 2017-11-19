#pragma once
#include <Base/Common.hpp>

CUDAInline glm::vec3 calcHalf(glm::vec3 in, glm::vec3 out) {
    return glm::normalize(in + out);
}

//Real Shading in Unreal Engine 4
CUDAInline vec3 fresnelSchlick(vec3 f0, float odh) {
    auto k = powf(2.0f, (-5.55473f*odh-6.98316f)*odh);
    return f0 + (1.0f - f0)*k;
}

CUDAInline float GGXD(float ndh, float roughness) {
    auto a = roughness*roughness;
    auto root = a / ((a*a-1.0f)*ndh*ndh+1.0f);
    return one_over_pi<float>()*root*root;
}

CUDAInline float GGXG(float ndo, float k) {
    auto div = ndo * (1.0f - k) + k;
    return ndo / div;
}

CUDAInline float smithG(float ndi, float ndo, float roughness) {
    auto alpha = roughness + 1.0f;
    auto k = alpha*alpha / 8.0f;
    return GGXG(ndi, k) * GGXG(ndo, k);
}

CUDAInline float lambertian() {
    return one_over_pi<float>();
}

//https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
CUDAInline float disneyDiffuse(float ndi, float ndo, float idh, float roughness) {
    auto cosi = 1.0f - fabs(ndi);
    auto cosi2 = cosi*cosi;
    auto coso = 1.0f - fabs(ndo);
    auto coso2 = coso*coso;
    auto fd90sub1 = 2.0f*idh*idh*roughness - 1.0f;
    return one_over_pi<float>()*(1.0f + fd90sub1*cosi*cosi2*cosi2)
        *(1.0f + fd90sub1*coso*coso2*coso2);
}

CUDAInline vec3 cookTorrance(float D, vec3 F, float G, float ndi, float ndo) {
    return F*(D*G / (4.0f*ndi*ndo+epsilon<float>()));
}
