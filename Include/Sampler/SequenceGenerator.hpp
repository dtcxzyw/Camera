#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

/*
Hammersley Sequence
by Holger Dammertz
http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
*/
CUDAINLINE float radicalInverse(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return bits * 2.3283064365386963e-10f; //0x100000000
}
CUDAINLINE vec2 hammersley(const unsigned int i, const unsigned int n) {
    return vec2(static_cast<float>(i) / static_cast<float>(n), radicalInverse(i));
}
