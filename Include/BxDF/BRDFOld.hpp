#pragma once
#include <Core/Common.hpp>
#include <Math/Geometry.hpp>
#include <Core/IncludeBegin.hpp>
#include <device_functions.h>
#include <Core/IncludeEnd.hpp>
#include <Spectrum/SpectrumConfig.hpp>

//F
DEVICEINLINE float fresnelSchlick(const float d) {
    auto x = deviceSaturate(1.0f - d);
    auto x2 = x * x;
    return x * x2 * x2;
}

//Real Shading in Unreal Engine 4
DEVICEINLINE float fresnelSchlickUE4(const float vdh) {
    return powf(2.0f, (-5.55473f * vdh - 6.98316f) * vdh);
}

/*
Artist Friendly Metallic Fresnel(JCGT, 2014)
http://jcgt.org/published/0003/04/03/
*/
DEVICEINLINE Spectrum fresnelGulbrandsen(const RGB& r, const RGB& g, const float u) {
    const auto nMin = (1.0f - r) / (1.0f + r);
    const auto sqrtr = sqrt(r);
    const auto nMax = (1.0f + sqrtr) / (1.0f - sqrtr);
    const auto n = mix(nMax, nMin, g);
    const auto na1 = n + 1.0f;
    const auto ns1 = n - 1.0f;
    const auto k2 = (na1 * na1 * r - ns1 * ns1) / (1.0f - r);
    const auto sum = n * n + k2;
    const auto n2u = 2.0f * u * n;
    const auto u2 = u * u;
    const auto sau2 = sum + u2;
    const auto smu2 = sum * u2 + 1.0f;
    const auto rs = (sau2 - n2u) / (sau2 + n2u);
    const auto rp = (smu2 - n2u) / (smu2 + n2u);
    return Spectrum{0.5f * (rs + rp)};
}

//D
DEVICEINLINE float GGXD(const float ndh, const float roughness) {
    const auto a = roughness * roughness;
    const auto root = a / ((a * a - 1.0f) * ndh * ndh + 1.0f);
    return one_over_pi<float>() * root * root;
}

DEVICEINLINE float DGTR2Aniso(const float ndh, const float ax, const float ay,
    const float xdh, const float ydh) {
    const auto d1 = xdh / ax;
    const auto d2 = ydh / ay;
    const auto k = d1 * d1 + d2 * d2 + ndh * ndh;
    const auto div = ax * ay * k * k;
    return one_over_pi<float>() / div;
}

DEVICEINLINE float DGTR1(const float ndh, const float alpha) {
    if (alpha >= 1.0f)return one_over_pi<float>();
    const auto sqra = alpha * alpha;
    const auto k = sqra - 1.0f;
    const auto t = 1.0f + k * ndh * ndh;
    return k / (pi<float>() * log(sqra) * t);
}

//G
DEVICEINLINE float GGXG(const float ndv, const float k) {
    const auto div = ndv * (1.0f - k) + k;
    return ndv / div;
}

DEVICEINLINE float smithGUE4(const float ndl, const float ndv, const float roughness) {
    const auto alpha = roughness + 1.0f;
    const auto k = alpha * alpha / 8.0f;
    return GGXG(ndl, k) * GGXG(ndv, k);
}

DEVICEINLINE float smithGGGX(const float ndv, const float alpha2) {
    const auto ndv2 = ndv * ndv;
    return 1.0f / (ndv + sqrt(ndv2 + alpha2 - ndv2 * alpha2));
}

DEVICEINLINE float smithGAniso(const float ndv, const float vdx, const float vdy,
    const float ax, const float ay) {
    const auto mx = vdx * ax;
    const auto my = vdy * ay;
    return 1.0f / (ndv + sqrt(mx * mx + my * my + ndv * ndv));
}

DEVICEINLINE float GLambda(const float u, const float alpha2) {
    const auto cosu2 = u * u;
    const auto sinu2 = 1.0f - cosu2;
    return (-1.0f + sqrt(1.0f + alpha2 * sinu2 / cosu2)) * 0.5f;
}

DEVICEINLINE float smithGHeightCorrelated(const float ndl, const float ndv, const float ldh,
    const float vdh, const float alpha2) {
    return fmin(ldh, vdh) > 0.0f ? 1.0f / (1.0f + GLambda(ndl, alpha2) + GLambda(ndv, alpha2)) : 0.0f;
}

//Diffuse
DEVICEINLINE float lambertian() {
    return one_over_pi<float>();
}

//https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
DEVICEINLINE float disneyDiffuse(const float ndl, const float ndv, const float ldh, const float roughness) {
    const auto cosi = 1.0f - ndl;
    const auto cosi2 = cosi * cosi;
    const auto coso = 1.0f - ndv;
    const auto coso2 = coso * coso;
    const auto fd90Sub1 = 2.0f * ldh * ldh * roughness - 0.5f;
    return one_over_pi<float>() * (1.0f + fd90Sub1 * cosi * cosi2 * cosi2)
        * (1.0f + fd90Sub1 * coso * coso2 * coso2);
}

DEVICEINLINE float disneyDiffuse2015(const float ndl, const float ndv, const float ldh, const float roughness) {
    const auto cosi = 1.0f - ndl;
    const auto cosi2 = cosi * cosi;
    const auto coso = 1.0f - ndv;
    const auto coso2 = coso * coso;
    const auto fl = cosi * cosi2 * cosi2;
    const auto fv = coso * coso2 * coso2;
    const auto rr = 2.0f * ldh * ldh * roughness;
    const auto lambert = (1.0f - 0.5f * fl) * (1.0f - 0.5f * fv);
    const auto retroReflection = rr * (fl + fv + fl * fv * (rr - 1.0f));
    return one_over_pi<float>() * (lambert + retroReflection);
}

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
    Spectrum baseColor; //linear color space
};

DEVICEINLINE Spectrum disneyBRDF(const Normal L, const Normal V, const Normal N, const Normal X,
    const Normal Y, const DisneyBRDFArg& arg) {
    const auto ndl = dot(L, N);
    const auto ndv = dot(V, N);
    if (fmin(ndl, ndv) < 0.0f)return Spectrum(0.0f);
    const auto H = halfVector(L, V);
    const auto ldh = dot(L, H);
    const auto ndh = dot(N, H);

    //sheen
    const auto lum = arg.baseColor.y();
    const auto ctint = arg.baseColor * (1.0f / lum);
    const auto cspec = mix(arg.specular * 0.08f * mix(Spectrum(1.0f), ctint, arg.specularTint), arg.baseColor,
        arg.metallic);
    const auto csheen = mix(Spectrum(1.0f), ctint, arg.sheenTint);
    const auto fh = fresnelSchlick(ldh);
    const auto sheen = fh * arg.sheen * csheen;

    //subsurface
    const auto fl = fresnelSchlick(ndl);
    const auto fv = fresnelSchlick(ndv);
    const auto fss90 = ldh * ldh * arg.roughness;
    const auto fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);
    const auto ss = 1.25f * (fss * (1.0f / (ndl + ndv) - 0.5f) + 0.5f);

    //diffuse fresnel
    const auto fd90 = 0.5f + 2.0f * fss90;
    const auto fd = mix(1.0f, fd90, fl) * mix(1.0f, fd90, fv);

    //specular
    const auto aspect = sqrt(1.0f - arg.anisotropic * 0.9f);
    const auto sqrr = arg.roughness * arg.roughness;
    const auto ax = fmax(0.001f, sqrr / aspect);
    const auto ay = fmax(0.001f, sqrr * aspect);
    const auto D = DGTR2Aniso(ndh, ax, ay, dot(X, H), dot(Y, H));
    const auto F = mix(cspec, Spectrum(1.0f), fh);
    const auto G = smithGAniso(ndl, dot(L, X), dot(L, Y), ax, ay) *
        smithGAniso(ndv, dot(V, X), dot(V, Y), ax, ay);
    const auto Vs = D * G * F;

    //clearcoat
    const auto Dc = DGTR1(ndh, mix(0.1f, 0.001f, arg.clearcoatGloss));
    const auto Fc = mix(0.04f, 1.0f, fh);
    constexpr auto a2 = 0.25f * 0.25f;
    const auto Gc = smithGGGX(ndl, a2) * smithGGGX(ndv, a2);
    const auto Vc = Dc * Fc * Gc;

    return (one_over_pi<float>() * mix(fd, ss, arg.subsurface) * arg.baseColor + sheen)
        * (1.0f - arg.metallic) + Vs + 0.25f * arg.clearcoat * Vc;
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
    Spectrum baseColor; //linear color space
};

//ratio = dis/radius
DEVICEINLINE Spectrum UE4BRDF(const Normal L, const Normal V, const Normal N, const Normal X, const Normal Y,
    const UE4BRDFArg& arg, const float ratio) {
    const auto ndl = dot(L, N);
    const auto ndv = dot(V, N);
    if (fmin(ndl, ndv) < 0.0f)return Spectrum(0.0f);
    const auto H = halfVector(L, V);
    const auto ldh = dot(L, H);
    const auto ndh = dot(N, H);

    //sheen
    const auto lum = arg.baseColor.y();
    const auto ctint = arg.baseColor * (1.0f - lum);
    const auto cspec = mix(arg.cavity * 0.08f * ctint, arg.baseColor, arg.metallic);
    const auto fh = fresnelSchlickUE4(ldh);
    const auto sheen = fh * arg.sheen * ctint;

    //subsurface
    const auto fl = fresnelSchlickUE4(ndl);
    const auto fv = fresnelSchlickUE4(ndv);
    const auto fss90 = ldh * ldh * arg.roughness;
    const auto fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);
    const auto ss = 1.25f * (fss * (1.0f / (ndl + ndv) - 0.5f) + 0.5f);

    //diffuse fresnel
    const auto fd = 1.0f;

    //specular
    const auto aspect = sqrt(1.0f - arg.anisotropy * 0.9f);
    const auto alpha = deviceSaturate(arg.roughness * arg.roughness + 0.5f * ratio);
    const auto ax = fmax(0.001f, alpha / aspect);
    const auto ay = fmax(0.001f, alpha * aspect);
    const auto D = DGTR2Aniso(ndh, ax, ay, dot(X, H), dot(Y, H));
    const auto F = mix(cspec, Spectrum(1.0f), fh);
    const auto G = smithGUE4(ndl, ndv, arg.roughness);
    const auto Vs = D * G * F;

    //clearcoat
    const auto Dc = DGTR1(ndh, 0.05f);
    const auto Fc = mix(0.04f, 1.0f, fh);
    constexpr auto a2 = 0.25f * 0.25f;
    const auto Gc = smithGGGX(ndl, a2) * smithGGGX(ndv, a2);
    const auto Vc = Dc * Fc * Gc;

    return (one_over_pi<float>() * mix(fd, ss, arg.subsurface) * arg.baseColor + sheen)
        * (1.0f - arg.metallic) + Vs + 0.25f * arg.clearcoat * Vc;
}

/*
Implementation of Frostbite BRDF ,as described in:
http://blog.selfshadow.com/publications/s2014-shading-course/frostbite/s2014_pbs_frostbite_slides.pdf
Diffuse:Disney's model
Specular:Microfacet model with GGX NDF
D:GGX
F:Schlick
G:Height-Correlated Smith[Heitz14]
http://jcgt.org/published/0003/02/03/
r->fv:
O                0    0          0
water          90  0.352   0.02
common    128 0.5       0.04
ruby           180 0.705   0.077
diamond    255 1          0.171
fv=0.2281399812785982*x^3-0.22673613288218408*x^2+0.20285923208572978*x-0.03326308048214361
f0=mix(fv,1.0,metallic)
*/
struct FrostbiteBRDFArg final {
    Spectrum baseColor;
    float metallic;
    float smoothness;
    float reflectance;
};

DEVICEINLINE float calcFv(const float r) {
    constexpr auto a = 0.2281399812785982f, b = -0.22673613288218408f
        , c = 0.20285923208572978f, d = -0.03326308048214361f;
    return deviceSaturate(((a * r + b) * r + c) * r + d);
}

DEVICEINLINE Spectrum frostbiteBRDF(const Normal L, const Normal V, const Normal N,
    const FrostbiteBRDFArg& arg) {
    const auto ndl = dot(L, N);
    const auto ndv = dot(V, N);
    if (fmin(ndl, ndv) < 0.0f)return Spectrum(0.0f);
    const auto H = halfVector(L, V);
    const auto ldh = dot(L, H);
    const auto ndh = dot(N, H);
    const auto vdh = dot(V, H);

    //diffuse
    const auto k = 1.0f - arg.smoothness;
    const auto roughness = k * k;
    const auto fd = arg.baseColor * (disneyDiffuse(ndl, ndv, ldh, roughness) * (1.0f - arg.metallic));

    //specular
    const auto D = GGXD(ndh, roughness);
    const auto f0 = mix(Spectrum(calcFv(arg.reflectance)), arg.baseColor, arg.metallic);
    const auto F = mix(f0, Spectrum(1.0f), fresnelSchlick(ldh));
    const auto G = smithGHeightCorrelated(ndl, ndv, ldh, vdh, roughness * roughness);
    const auto fs = F * (D * G / (4.0f * ndl * ndv));
    return fd + fs;
}

/*
mixed BRDF
This BRDF depends on http://blog.selfshadow.com/publications/s2016-shading-course/hoffman/s2016_pbs_recent_advances_v2.pdf.
Diffuse:Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering¡±, Burley, SIGGRAPH Course Notes 2015
http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
F:Artist Friendly Metallic Fresnel (JCGT, 2014)
http://jcgt.org/published/0003/04/03/
D:Anisotropic GGX
http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
G:Height-Correlated Smith[Heitz14]
http://jcgt.org/published/0003/02/03/
*/
struct MixedBRDFArg final {
    Spectrum baseColor;
    Spectrum edgeTint;
    float metallic;
    float roughness;
    float anisotropic;
};

DEVICEINLINE Spectrum mixedBRDF(const Normal L, const Normal V, const Normal N, const Normal X, const Normal Y,
    const MixedBRDFArg& arg) {
    const auto ndl = dot(L, N);
    const auto ndv = dot(V, N);
    if (fmin(ndl, ndv) < 0.0f)return Spectrum(0.0f);
    const auto H = halfVector(L, V);
    const auto ldh = dot(L, H);
    const auto vdh = dot(V, H);
    const auto ndh = dot(N, H);

    const auto fd = arg.baseColor * (disneyDiffuse2015(ndl, ndv, ldh, arg.roughness) * (1.0f - arg.metallic));
    const auto F = fresnelGulbrandsen(arg.baseColor.toRGB(), arg.edgeTint.toRGB(), ldh);
    const auto aspect = sqrt(1.0f - arg.anisotropic * 0.9f);
    const auto sqrr = arg.roughness * arg.roughness;
    const auto ax = fmax(0.001f, sqrr / aspect);
    const auto ay = fmax(0.001f, sqrr * aspect);
    const auto kx = ax * dot(X, V), ky = ay * dot(Y, V);
    const auto G = smithGHeightCorrelated(ndl, ndv, ldh, vdh, kx * kx + ky * ky);
    const auto D = DGTR2Aniso(ndh, ax, ay, dot(X, H), dot(Y, H));
    const auto fs = F * (D * G / (4.0f * ndl * ndv));
    return fd + fs;
}
