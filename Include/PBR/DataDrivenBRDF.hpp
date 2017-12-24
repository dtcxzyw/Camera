#pragma once
#include <Base/Common.hpp>
#include <Base/Memory.hpp>

/*
http://www.merl.com/brdf/

"A Data-Driven Reflectance Model",
Wojciech Matusik, Hanspeter Pfister, Matt Brand and Leonard McMillan,
ACM Transactions on Graphics 22, 3(2003), 759-769.
BibTeX:
@article {Matusik:2003,
    author = "Wojciech Matusik and Hanspeter Pfister and Matt Brand and Leonard McMillan",
    title = "A Data-Driven Reflectance Model",
    journal = "ACM Transactions on Graphics",
    year = "2003",
    month = jul,
    volume = "22",
    number = "3",
    pages = "759-769"
}

The implementation is based on http://people.csail.mit.edu/wojciech/BRDFDatabase/code/BRDFRead.cpp.
*/

constexpr auto rth = 90, rtd = 90, rpd = 360, size = rth *rtd *rpd / 2;
constexpr auto rfac = 1.0f / 1500.0f,gfac = 1.15f / 1500.0f,bfac=1.66f / 1500.0f;

namespace Impl {

    // rotate vector along one axis
    CUDAInline vec3 rotateVector(vec3 vector, vec3 axis, float angle) {
        float cosAng = cos(angle);
        float sinAng = sin(angle);
        vec3 out = vector * cosAng;
        out += axis * dot(axis, vector)*(1.0f - cosAng);
        out += glm::cross(axis, vector) * sinAng;
        return out;
    }

    // convert standard coordinates to half vector/difference vector coordinates
    CUDAInline void std2half(vec3 in, vec3 out, vec3 half, vec3 normal,vec3 bin,
        float& thalf, float& phalf, float& tdiff, float& pdiff) {

        // compute  theta_half, fi_half
        thalf = acos(half[2]);
        phalf = atan2(half[1], half[0]);

        // compute diff vector
        vec3 temp=rotateVector(in, normal, -phalf);
        vec3 diff=rotateVector(temp, bin, -thalf);

        // compute  theta_diff, fi_diff	
        tdiff = acos(diff[2]);
        pdiff = atan2(diff[1], diff[0]);
    }

    // Lookup theta_half index
    // This is a non-linear mapping!
    // In:  [0 .. pi/2]
    // Out: [0 .. 89]
    CUDAInline int thid(float thalf) {
        int res =rth*sqrt(fmax(thalf,0.0f) / half_pi<float>());
        return clamp(res,0,rth-1);
    }

    // Lookup theta_diff index
    // In:  [0 .. pi/2]
    // Out: [0 .. 89]
    CUDAInline int tdid(float tdiff) {
        return clamp(static_cast<int>(tdiff / half_pi<float>() * rtd),0,rtd-1);
    }

    // Lookup phi_diff index
    // In: phi_diff in [0 .. pi]
    // Out: tmp in [0 .. 179]
    CUDAInline int pdid(float pdiff) {
        // Because of reciprocity, the BRDF is unchanged under
        // phi_diff -> phi_diff + M_PI
        if (pdiff < 0.0f)pdiff += pi<float>();
        return clamp(static_cast<int>(pdiff / two_pi<float>() * rpd), 0, rpd / 2 - 1);
    }

    // Given a pair of incoming/outgoing angles, look up the BRDF.
    CUDAInline int lookupBRDF(vec3 in, vec3 out, vec3 half, vec3 normal,vec3 bin) {
        // Convert to halfangle / difference angle coordinates
        float thalf, phalf, tdiff, pdiff;

        std2half(in,out,half,normal,bin,thalf, phalf, tdiff, pdiff);

        // Find index.
        // Note that phi_half is ignored, since isotropic BRDFs are assumed
        return pdid(pdiff) +tdid(tdiff) * (rpd / 2) + thid(thalf) * (rpd / 2 *rtd);
    }
}

class BRDFSampler final {
private:
    const vec3* ReadOnlyCache mData;
public:
    BRDFSampler() = default;
    BRDFSampler(const vec3* ReadOnlyCache data);
    CUDAInline RGB get(vec3 in, vec3 out,vec3 half,vec3 normal,vec3 bin) const {
        return mData[Impl::lookupBRDF(in, out, half, normal,bin)];
    }
};

class MERLBRDFData final {
private:
    DataViewer<vec3> mData;
public:
    MERLBRDFData(const std::string& path);
    BRDFSampler toSampler() const;
};
