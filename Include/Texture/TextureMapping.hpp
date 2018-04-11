#pragma once
#include <Base/Common.hpp>
#include <Base/Math.hpp>

//The trick comes from https://learnopengl.com/#!PBR/IBL/Diffuse-irradiance.
CUDAINLINE vec2 sphericalMapping(const vec3 p) {
    return { atan(p.z, p.x) * one_over_two_pi<float>() + 0.5f, asin(p.y) * one_over_pi<float>() + 0.5f };
}

CUDAINLINE vec2 cylindricalMapping(const vec3 p) {
    return { atan2(p.z,p.x) * one_over_two_pi<float>() + 0.5f ,p.y };
}

CUDAINLINE vec2 planarMapping(const vec3 p,const vec3 s,const vec3 t) {
    return { dot(p,s),dot(p,t) };
}
