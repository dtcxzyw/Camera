#pragma once
#include <Base/Common.hpp>

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
constexpr auto rfac = 1.0f / 1500.0f;
constexpr auto gfac = 1.15f / 1500.0f;
constexpr auto bfac = 1.66f / 1500.0f;

namespace Impl {

    // rotate vector along one axis
    CUDAInline vec3 rotate_vector(vec3 vector, vec3 axis, float angle) {
        float cos_ang = cos(angle);
        float sin_ang = sin(angle);
        vec3 out = vector * cos_ang;
        out += axis * dot(axis, vector)*(1.0f - cos_ang);
        out += glm::cross(axis, vector) * sin_ang;
        return out;
    }

    // convert standard coordinates to half vector/difference vector coordinates
    CUDAInline void std_coords_to_half_diff_coords(float theta_in, float fi_in, float theta_out, float fi_out,
        float& theta_half, float& fi_half, float& theta_diff, float& fi_diff) {

        // compute in vector
        float in_vec_z = cos(theta_in);
        float proj_in_vec = sin(theta_in);
        float in_vec_x = proj_in_vec*cos(fi_in);
        float in_vec_y = proj_in_vec*sin(fi_in);
        auto in= normalize(vec3{ in_vec_x,in_vec_y,in_vec_z });

        // compute out vector
        float out_vec_z = cos(theta_out);
        float proj_out_vec = sin(theta_out);
        float out_vec_x = proj_out_vec*cos(fi_out);
        float out_vec_y = proj_out_vec*sin(fi_out);
        auto out =normalize(vec3({ out_vec_x,out_vec_y,out_vec_z }));

        // compute halfway vector
        float half_x = (in_vec_x + out_vec_x) / 2.0f;
        float half_y = (in_vec_y + out_vec_y) / 2.0f;
        float half_z = (in_vec_z + out_vec_z) / 2.0f;
        vec3 half =normalize(vec3({ half_x,half_y,half_z }));

        // compute  theta_half, fi_half
        theta_half = acos(half[2]);
        fi_half = atan2(half[1], half[0]);

        vec3 bi_normal = { 0.0f, 1.0f, 0.0f };
        vec3 normal = { 0.0f, 0.0f, 1.0f };

        // compute diff vector
        vec3 temp=rotate_vector(in, normal, -fi_half);
        vec3 diff=rotate_vector(temp, bi_normal, -theta_half);

        // compute  theta_diff, fi_diff	
        theta_diff = acos(diff[2]);
        fi_diff = atan2(diff[1], diff[0]);
    }

    // Lookup theta_half index
    // This is a non-linear mapping!
    // In:  [0 .. pi/2]
    // Out: [0 .. 89]
    CUDAInline int theta_half_index(float theta_half) {
        if (theta_half <= 0.0f)return 0;
        int res =rth*sqrt(theta_half / half_pi<float>());
        return clamp(res,0,rth-1);
    }

    // Lookup theta_diff index
    // In:  [0 .. pi/2]
    // Out: [0 .. 89]
    CUDAInline int theta_diff_index(float theta_diff) {
        return clamp(static_cast<int>(theta_diff / half_pi<float>() * rtd),0,rtd-1);
    }

    // Lookup phi_diff index
    // In: phi_diff in [0 .. pi]
    // Out: tmp in [0 .. 179]
    CUDAInline int phi_diff_index(float phi_diff) {
        // Because of reciprocity, the BRDF is unchanged under
        // phi_diff -> phi_diff + M_PI
        if (phi_diff < 0.0f)phi_diff += pi<float>();
        return clamp(static_cast<int>(phi_diff / two_pi<float>() * rpd), 0, rpd / 2 - 1);
    }

    // Given a pair of incoming/outgoing angles, look up the BRDF.
    CUDAInline int lookup_brdf_idx(float theta_in, float fi_in,float theta_out, float fi_out) {
        // Convert to halfangle / difference angle coordinates
        float theta_half, fi_half, theta_diff, fi_diff;

        std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out,
            theta_half, fi_half, theta_diff, fi_diff);

        // Find index.
        // Note that phi_half is ignored, since isotropic BRDFs are assumed
        return phi_diff_index(fi_diff) +
            theta_diff_index(theta_diff) * (rpd / 2) +
            theta_half_index(theta_half) * (rpd / 2 *rtd);
    }

    CUDAInline vec2 calcAngle(vec3 vec) {
        vec2 v2 = { vec.x,vec.z };
        auto len = length(v2);
        auto theta = acos(len);
        auto phi = dot(v2/len, vec2{ 0.0f,1.0f });
        if (v2.x < 0.0f)phi += pi<float>();
        return { theta,phi };
    }
}

class BRDFSampler final {
private:
    const vec3* ReadOnly mData;
public:
    BRDFSampler(const vec3* ReadOnly data);
    CUDAInline RGB get(vec3 in, vec3 normal, vec3 out) const {
        //normal-space
        auto rotate = rotation(normal, { 0.0f,1.0f,0.0f });
        auto inn=normalize(in*rotate),outn=normalize(out*rotate);
        auto ain = Impl::calcAngle(inn), aout = Impl::calcAngle(outn);
        int idx=Impl::lookup_brdf_idx(ain.x,ain.y,aout.x,aout.y);
        return mData[idx];
    }
};

class MERLBRDFData final {
private:
    DataViewer<vec3> mData;
public:
    MERLBRDFData(const std::string& path);
    BRDFSampler toSampler() const;
};
