#pragma once
#include <Core/Memory.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Core/Builtin.hpp>
#include <Texture/FilterWrapper.hpp>

class FilmTileRef final {
private:
    Spectrum* mPixel;
    float* mWeightSum;
    unsigned int mWidth;
    BuiltinSamplerRef<float> mWeightLUT;
public:
    FilmTileRef(Spectrum* pixel, float* weightSum,
        uvec2 size, const BuiltinSamplerRef<float>& weightLUT);
    DEVICE void add(vec2 pos, const Spectrum& spectrum);
};

class SampleWeightLUT final {
private:
    std::shared_ptr<BuiltinArray<float>> mArray;
    std::shared_ptr<BuiltinSampler<float>> mSampler;
    void cookTable(const FilterWrapper& filter);
public:
    explicit SampleWeightLUT(unsigned int tableSize, const FilterWrapper& filter);
    BuiltinSamplerRef<float> toRef() const;
};

class FilmTile final : Uncopyable {
private:
    Span<Spectrum> mPixel;
    Span<float> mWeight;
    SampleWeightLUT mWeightLUT;
    uvec2 mSize;
    CommandBuffer& mBuffer;
public:
    explicit FilmTile(CommandBuffer& buffer, uvec2 size, const SampleWeightLUT& weightLUT);
    void merge(const MemorySpan<Spectrum>& pixel, const MemorySpan<float>& weight,
        uvec2 dstSize, uvec2 offset) const;
    uvec2 size() const;

    auto toRef() const {
        return mBuffer.makeLazyConstructor<FilmTileRef>(mPixel, mWeight, mSize, mWeightLUT.toRef());
    }
};
