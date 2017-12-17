#pragma once
#include <Base/Common.hpp>
#include <device_functions.h>

CUDAInline vec3 calcHalf(vec3 in, vec3 out) {
    return normalize(in + out);
}

//F
CUDAInline float fresnelSchlick(float d) {
    auto x =saturate(1.0f - d);
    auto x2 = x * x;
    return x*x2*x2;
}

//Real Shading in Unreal Engine 4
CUDAInline float fresnelSchlickUE4(float vdh) {
    return powf(2.0f, (-5.55473f*vdh-6.98316f)*vdh);
}

//D
CUDAInline float GGXD(float ndh, float roughness) {
    auto a = roughness*roughness;
    auto root = a / ((a*a-1.0f)*ndh*ndh+1.0f);
    return one_over_pi<float>()*root*root;
}

CUDAInline float DGTR2Aniso(float ndh, float ax,float ay,float xdh,float ydh) {
    auto d1 = xdh/ ax,d2=ydh/ay;
    auto k = d1*d1+d2*d2+ ndh * ndh;
    auto div = ax*ay*k*k;
    return one_over_pi<float>() /div;
}

CUDAInline float DGTR1(float ndh, float alpha) {
    if (alpha >= 1.0f)return one_over_pi<float>();
    float sqra = alpha * alpha;
    float k = sqra - 1.0f;
    float t = 1.0f +k*ndh*ndh;
    return k / (pi<float>()*log(sqra)*t);
}

//G
CUDAInline float GGXG(float ndv, float k) {
    auto div = ndv * (1.0f - k) + k;
    return ndv / div;
}

CUDAInline float smithGUE4(float ndl, float ndv, float roughness) {
    auto alpha = roughness + 1.0f;
    auto k = alpha*alpha / 8.0f;
    return GGXG(ndl, k) * GGXG(ndv, k);
}

CUDAInline float smithGGGX(float ndv, float alpha2) {
    auto ndv2 = ndv * ndv;
    return 1.0f / (ndv+sqrt(ndv2+alpha2-ndv2*alpha2));
}

CUDAInline float smithGAniso(float ndv, float vdx,float vdy, float ax,float ay) {
    auto mx = vdx * ax, my = vdy * ay;
    return 1.0f / (ndv +sqrt(mx*mx+my*my+ndv*ndv));
}

//Diffuse
CUDAInline float lambertian() {
    return one_over_pi<float>();
}

//https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
CUDAInline float disneyDiffuse(float ndl, float ndv, float ldh, float roughness) {
    auto cosi = 1.0f - fabs(ndl);
    auto cosi2 = cosi*cosi;
    auto coso = 1.0f - fabs(ndv);
    auto coso2 = coso*coso;
    auto fd90sub1 = 2.0f*ldh*ldh*roughness - 0.5f;
    return one_over_pi<float>()*(1.0f + fd90sub1*cosi*cosi2*cosi2)
        *(1.0f + fd90sub1*coso*coso2*coso2);
}

//BRDF
template<typename... Arg>
using BRDF = vec3(*)(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y, Arg... arg);

/*
Implementation of Disney "principled" BRDF ,as described in:
http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf

The implementation is based on https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf,
which is licensed under the Apache License, Version 2.0.
*/
struct DisneyBRDFArg final {
    float metallic;
    float subsurface;
    float specular;
    float roughness;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    vec3 baseColor;//linear color space
};

CUDAInline vec3 disneyBRDF(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y,DisneyBRDFArg arg) {
    auto ndl = dot(L, N);
    auto ndv = dot(V, N);
    if (fmin(ndl, ndv) < 0.0f)return vec3(0.0f);
    auto H = calcHalf(L, V);
    auto ldh = dot(L, H);
    auto ndh = dot(N,H);

    //sheen
    auto lum = luminosity(arg.baseColor);
    auto ctint = lum > 0.0f ? arg.baseColor / lum : vec3(1.0f);
    auto cspec = mix(arg.specular*0.08f*mix(vec3(1.0f), ctint, arg.specularTint), arg.baseColor, arg.metallic);
    auto csheen = mix(vec3(1.0f), ctint,arg.sheenTint);
    auto fh = fresnelSchlick(ldh);
    auto sheen = fh * arg.sheen * csheen;

    //subsurface
    auto fl = fresnelSchlick(ndl), fv = fresnelSchlick(ndv);
    auto fss90 = ldh * ldh*arg.roughness;
    auto fss = mix(1.0f, fss90, fl)*mix(1.0f, fss90, fv);
    auto ss = 1.25f * (fss * (1.0f / (ndl + ndv) - 0.5f) + 0.5f);

    //diffuse fresnel
    auto fd90 = 0.5f + 2.0f*fss90;
    auto fd = mix(1.0f, fd90, fl)*mix(1.0f, fd90, fv);

    //specular
    auto aspect = sqrt(1.0f - arg.anisotropic*0.9f);
    auto sqrr = arg.roughness*arg.roughness;
    auto ax = fmax(0.001f, sqrr*aspect);
    auto ay = fmax(0.001f, sqrr / aspect);
    auto D = DGTR2Aniso(ndh, ax, ay,dot(X,H),dot(Y,H));
    auto F = mix(cspec,vec3(1.0f),fh);
    auto G = smithGAniso(ndl, dot(L, X), dot(L, Y), ax, ay)*
        smithGAniso(ndv,dot(V,X),dot(V,Y),ax,ay);
    auto Vs = D * G * F;

    //clearcoat
    auto Dc = DGTR1(ndh,mix(0.1f,0.001f,arg.clearcoatGloss));
    auto Fc = mix(0.04f, 1.0f, fh);
    constexpr auto a2 = 0.25f*0.25f;
    auto Gc = smithGGGX(ndl,a2)*smithGGGX(ndv,a2);
    auto Vc = Dc * Fc*Gc;

    return (one_over_pi<float>()*mix(fd, ss, arg.subsurface)*arg.baseColor + sheen)
        *(1.0f - arg.metallic) + Vs + 0.25f*arg.clearcoat*Vc;
}

/*
Implementation of UE4 BRDF ,as described in:
http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
The implementation may be wrong because I couldn't learn more from this paper.
*/
struct UE4BRDFArg final {
    float metallic;
    float roughness;
    float cavity;
    float subsurface;
    float anisotropy;
    float clearcoat;
    float sheen;
    vec3 baseColor;//linear color space
};
//ratio = dis/radius
CUDAInline vec3 UE4BRDF(vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y, UE4BRDFArg arg,float ratio) {
    auto ndl = dot(L, N);
    auto ndv = dot(V, N);
    if (fmin(ndl, ndv) < 0.0f)return vec3(0.0f);
    auto H = calcHalf(L, V);
    auto ldh = dot(L, H);
    auto ndh = dot(N, H);

    //sheen
    auto lum = luminosity(arg.baseColor);
    auto ctint = lum > 0.0f ? arg.baseColor / lum : vec3(1.0f);
    auto cspec = mix(arg.cavity*0.08f*ctint, arg.baseColor, arg.metallic);
    auto fh = fresnelSchlickUE4(ldh);
    auto sheen = fh * arg.sheen * ctint;

    //subsurface
    auto fl = fresnelSchlickUE4(ndl), fv = fresnelSchlickUE4(ndv);
    auto fss90 = ldh * ldh*arg.roughness;
    auto fss = mix(1.0f, fss90, fl)*mix(1.0f, fss90, fv);
    auto ss = 1.25f * (fss * (1.0f / (ndl + ndv) - 0.5f) + 0.5f);

    //diffuse fresnel
    auto fd = lambertian();

    //specular
    auto aspect = sqrt(1.0f - arg.anisotropy*0.9f);
    auto alpha = saturate(arg.roughness*arg.roughness+0.5f*ratio);
    auto ax = fmax(0.001f, alpha*aspect);
    auto ay = fmax(0.001f, alpha / aspect);
    auto D = DGTR2Aniso(ndh, ax, ay, dot(X, H), dot(Y, H));
    auto F = mix(cspec, vec3(1.0f), fh);
    auto G = smithGUE4(ndl,ndv,arg.roughness);
    auto Vs = D * G * F;

    //clearcoat
    auto Dc = DGTR1(ndh, 0.1f);
    auto Fc = mix(0.04f, 1.0f, fh);
    constexpr auto a2 = 0.25f*0.25f;
    auto Gc = smithGGGX(ndl, a2)*smithGGGX(ndv, a2);
    auto Vc = Dc * Fc*Gc;

    return (one_over_pi<float>()*mix(fd, ss, arg.subsurface)*arg.baseColor + sheen)
        *(1.0f - arg.metallic) + Vs + 0.25f*arg.clearcoat*Vc;
}
