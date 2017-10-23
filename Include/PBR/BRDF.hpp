#pragma once
#include <Base/Common.hpp>

CUDAInline glm::vec3 calcHalf(glm::vec3 in, glm::vec3 out) {
    return glm::normalize(in + out);
}

CUDAInline vec3 fresnelSchlick(vec3 f0, float idh) {
    float sidh = 1.0f - idh;
    float sidh2 = sidh*sidh;
    return f0 + (1.0f - f0)*(sidh*sidh2*sidh2);
}

CUDAInline float GGXD(float ndh, float a) {
    float snh = sin(acos(ndh));
    float root = a / (a*a*ndh*ndh + snh*snh);
    return one_over_pi<float>()*root*root;
}

CUDAInline float GGXG(float ndo, float a) {
    auto k = a*a / 8.0f;
    auto div = ndo * (1.0f - k) + k;
    return ndo / div;
}

CUDAInline float smithG(float ndi, float ndo, float a) {
    return GGXG(ndi, a) * GGXG(ndo, a);
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

CUDAInline vec3 cookTorrance(float diff, float D, vec3 F, float G, float ndi, float ndo) {
    return diff + F*(D*G / (4.0f*ndi*ndo+epsilon<float>()));
}
