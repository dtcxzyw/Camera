#pragma once
#include <Core/Memory.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Core/Builtin.hpp>
#include <Texture/FilterWrapper.hpp>

class FilmTileRef final {
private:
    Spectrum* mPixel;
    unsigned int mWidth;
    BuiltinSamplerRef<float> mWeight;
public:
    FilmTileRef(const MemorySpan<Spectrum> pixel, const unsigned int width,
        const BuiltinSamplerRef<float>& weight)
        :mPixel(pixel.begin()), mWidth(width), mWeight(weight) {}
    DEVICE void add(vec2 pos, const Spectrum& spectrum);
};

class SampleWeightLUT {
private:
    std::shared_ptr<BuiltinArray<float>> mArray;
    std::shared_ptr<BuiltinSampler<float>> mSampler;
    void cookTable(const FilterWrapper& filter);
public:
    explicit SampleWeightLUT(const unsigned int tableSize, const FilterWrapper& filter):
        mArray(std::make_shared<BuiltinArray<float>>(vec2{ tableSize,tableSize })),
        mSampler(std::make_shared<BuiltinSampler<float>>(mArray->get(), cudaAddressModeBorder)) {
        cookTable(filter);
    }
    BuiltinSamplerRef<float> toRef() const {
        return mSampler->toRef();
    }
};

class FilmTile final {
private:
    MemorySpan<Spectrum> mPixel;
    SampleWeightLUT mWeight;
    uvec2 mSize;
public:
    explicit FilmTile(const uvec2 size,const SampleWeightLUT& weight)
        :mPixel(size.x*size.y), mWeight(weight), mSize(size) {}
    void download(CommandBuffer& buffer, PinnedBuffer<Spectrum>& dst,
        uvec2 dstSize, uvec2 offset) const;
    uvec2 size() const {
        return mSize;
    }
    FilmTileRef toRef() const {
        return { mPixel,mSize.x ,mWeight.toRef() };
    }
};
