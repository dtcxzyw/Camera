#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

struct SequenceGenerator1DTag {};

struct SequenceGenerator2DTag {};

DEVICEINLINE float scaleToFloat(const unsigned int val) {
    union Result {
        unsigned u;
        float f;
    } result; // Write reversed bits directly into floating-point mantissa.
    result.u = 0x3f800000u | (val >> 9);
    return result.f - 1.0f;
}

DEVICEINLINE float radicalInverse(unsigned int index) {
    index = (index << 16u) | (index >> 16u);
    index = ((index & 0x55555555u) << 1u) | ((index & 0xAAAAAAAAu) >> 1u);
    index = ((index & 0x33333333u) << 2u) | ((index & 0xCCCCCCCCu) >> 2u);
    index = ((index & 0x0F0F0F0Fu) << 4u) | ((index & 0xF0F0F0F0u) >> 4u);
    index = ((index & 0x00FF00FFu) << 8u) | ((index & 0xFF00FF00u) >> 8u);
    return scaleToFloat(index);
}

class RadicalInverse final : SequenceGenerator1DTag {
public:
    DEVICE float sample(const unsigned int index) const {
        return radicalInverse(index);
    }
};

/*
Hammersley Sequence
by Holger Dammertz
http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
*/

class Hammersley2D final /*: SequenceGenerator2DTag*/ {
private:
    float mInvSize;
public:
    explicit Hammersley2D(const unsigned int size) : mInvSize(1.0f / size) {}
    DEVICE vec2 sample(const unsigned int index) const {
        return {static_cast<float>(index) * mInvSize, radicalInverse(index)};
    }
};

/*
Scrambled Halton Sequence
by Leonhard
http://gruenschloss.org/
 */
DEVICEINLINE float halton3(const unsigned int index) {
    constexpr unsigned int perm3[243] = {
        0, 81, 162, 27, 108, 189, 54, 135, 216, 9, 90, 171, 36, 117, 198, 63, 144, 225, 18, 99, 180, 45, 126, 207, 72,
        153, 234, 3, 84, 165, 30, 111, 192, 57, 138, 219, 12, 93, 174, 39, 120, 201, 66, 147, 228, 21, 102, 183, 48,
        129, 210, 75, 156, 237, 6, 87, 168, 33, 114, 195, 60, 141, 222, 15, 96, 177, 42, 123, 204, 69, 150, 231, 24,
        105, 186, 51, 132, 213, 78, 159, 240, 1, 82, 163, 28, 109, 190, 55, 136, 217, 10, 91, 172, 37, 118, 199, 64,
        145, 226, 19, 100, 181, 46, 127, 208, 73, 154, 235, 4, 85, 166, 31, 112, 193, 58, 139, 220, 13, 94, 175, 40,
        121, 202, 67, 148, 229, 22, 103, 184, 49, 130, 211, 76, 157, 238, 7, 88, 169, 34, 115, 196, 61, 142, 223, 16,
        97, 178, 43, 124, 205, 70, 151, 232, 25, 106, 187, 52, 133, 214, 79, 160, 241, 2, 83, 164, 29, 110, 191, 56,
        137, 218, 11, 92, 173, 38, 119, 200, 65, 146, 227, 20, 101, 182, 47, 128, 209, 74, 155, 236, 5, 86, 167, 32,
        113, 194, 59, 140, 221, 14, 95, 176, 41, 122, 203, 68, 149, 230, 23, 104, 185, 50, 131, 212, 77, 158, 239, 8,
        89, 170, 35, 116, 197, 62, 143, 224, 17, 98, 179, 44, 125, 206, 71, 152, 233, 26, 107, 188, 53, 134, 215, 80,
        161, 242
    };
    return (perm3[index % 243u] * 14348907u +
        perm3[(index / 243u) % 243u] * 59049u +
        perm3[(index / 59049u) % 243u] * 243u +
        perm3[(index / 14348907u) % 243u]) * 2.8679716489035376e-10f; // Results in [0,1).
}

class Halton2D final : SequenceGenerator2DTag {
public:
    DEVICE vec2 sample(const unsigned int index) const {
        return {radicalInverse(index), halton3(index)};
    }
};

//Generator matrix from pbrt-v3
//https://github.com/mmp/pbrt-v3/blob/master/src/core/sobolmatrices.cpp
DEVICEINLINE float scrambledSobol(const unsigned int index, const unsigned int dim,
    const unsigned int scramble) {
    constexpr unsigned int mat[2][32] = {
        {
            0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000,
            0x02000000, 0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000,
            0x00080000, 0x00040000, 0x00020000, 0x00010000, 0x00008000, 0x00004000,
            0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100,
            0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004,
            0x00000002, 0x00000001
        },
        {
            0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000,
            0xaa000000, 0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
            0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000,
            0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
            0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc,
            0xaaaaaaaa, 0xffffffff
        }
    };

    auto res = scramble;
    #pragma unroll
    for (auto i = 0; i < 32; ++i)
        res ^= (index & (1U << i) ? 0xffffffff : 0) & mat[dim][i];
    return scaleToFloat(res);
}

class Sobol1D final : SequenceGenerator1DTag {
private:
    const unsigned int mScramble;
public:
    explicit Sobol1D(const unsigned int scramble) : mScramble(scramble) {}
    DEVICE float sample(const unsigned int index) const {
        return scrambledSobol(index, 0, mScramble);
    }
};

class Sobol2D final : SequenceGenerator2DTag {
private:
    const unsigned int mScramble;
public:
    explicit Sobol2D(const unsigned int scramble) : mScramble(scramble) {}
    DEVICE vec2 sample(const unsigned int index) const {
        return {scrambledSobol(index, 0, mScramble), scrambledSobol(index, 1, mScramble)};
    }
};
