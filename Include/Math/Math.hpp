#pragma once
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CTOR_INIT
#include <Core/CompileBegin.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <Core/CompileEnd.hpp>
#include <Core/Common.hpp>

using glm::vec2;
using glm::uvec2;
using glm::ivec2;
using glm::uvec3;
using glm::vec4;
using glm::uvec4;
using glm::mat3;

using glm::max;
using glm::min;
using glm::clamp;
using glm::mix;

using glm::pi;
using glm::epsilon;
using glm::two_pi;
using glm::two_over_pi;
using glm::one_over_pi;
using glm::one_over_two_pi;
using glm::half_pi;
using glm::quarter_pi;

using Vector = glm::vec3;
using UV = vec2;
using RGB = glm::vec3;
using RGBA = vec4;
using RGB8 = glm::tvec3<unsigned char>;
using RGBA8 = glm::tvec4<unsigned char>;
using A8 = unsigned char;

constexpr auto maxv = std::numeric_limits<unsigned int>::max();

DEVICEINLINE float max3(const float a, const float b, const float c) {
    return fmax(a, fmax(b, c));
}

DEVICEINLINE float absMax3(const float a, const float b, const float c) {
    return max3(fabs(a), fabs(b), fabs(c));
}

DEVICEINLINE float min3(const float a, const float b, const float c) {
    return fmin(a, fmin(b, c));
}
