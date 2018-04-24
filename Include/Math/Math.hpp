#pragma once
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <Core/CompileBegin.hpp>
#include <glm/glm.hpp>
#include <Core/CompileEnd.hpp>
using namespace glm;

using UV = vec2;
using RGB = vec3;
using RGBA = vec4;
using RGB8 = tvec3<unsigned char>;
using RGBA8 = tvec4<unsigned char>;
using A8 = unsigned char;

constexpr auto maxv = std::numeric_limits<unsigned int>::max();
