#pragma once
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/color_space.hpp>
#include <glm/gtx/color_space.hpp>
using namespace glm;

using UV = vec2;
using RGB = vec3;
using RGBA = vec4;
using RGB8 = tvec3<unsigned char>;
using RGBA8 = tvec4<unsigned char>;
