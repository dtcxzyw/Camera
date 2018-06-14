#pragma once
#include <Core/Common.hpp>
#include <Math/Math.hpp>

struct SequenceGenerator1DTag {};

struct SequenceGenerator2DTag {};

DEVICE float scaleToFloat(uint32_t val);

DEVICE float radicalInverse2(uint32_t index);

class RadicalInverse final : SequenceGenerator1DTag {
public:
    DEVICE float sample(const uint32_t index) const {
        return radicalInverse2(index);
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
    explicit Hammersley2D(const uint32_t size) : mInvSize(1.0f / size) {}
    DEVICE vec2 sample(const uint32_t index) const {
        return {static_cast<float>(index) * mInvSize, radicalInverse2(index)};
    }
};

/*
Scrambled Halton Sequence
by Leonhard
http://gruenschloss.org/
 */
DEVICE float halton3(uint32_t index);
DEVICE float halton5(uint32_t index);
DEVICE float halton7(uint32_t index);
DEVICE float halton11(uint32_t index);

class Halton2D final : SequenceGenerator2DTag {
public:
    DEVICE vec2 sample(const uint32_t index) const {
        return {radicalInverse2(index), halton3(index)};
    }
};

//Generator matrix is from pbrt-v3
//https://github.com/mmp/pbrt-v3/blob/master/src/core/sobolmatrices.cpp
DEVICE float scrambledSobol(uint32_t index, uint32_t dim, uint32_t scramble);

class Sobol1D final : SequenceGenerator1DTag {
private:
    const uint32_t mScramble;
public:
    explicit Sobol1D(const uint32_t scramble) : mScramble(scramble) {}
    DEVICE float sample(const uint32_t index) const {
        return scrambledSobol(index, 0, mScramble);
    }
};

class Sobol2D final : SequenceGenerator2DTag {
private:
    const uint32_t mScramble;
public:
    explicit Sobol2D(const uint32_t scramble) : mScramble(scramble) {}
    DEVICE vec2 sample(const uint32_t index) const {
        return {scrambledSobol(index, 0, mScramble), scrambledSobol(index, 1, mScramble)};
    }
};
