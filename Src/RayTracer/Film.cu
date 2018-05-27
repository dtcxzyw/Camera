#include <RayTracer/Film.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>

GLOBAL void cookKernel(const unsigned int size, BuiltinRenderTargetRef<float> dst,
    FilterWrapper filter, const unsigned int width) {
    const auto id = getId();
    if (id >= size)return;
    const ivec2 pos = { id / width ,id%width };
    dst.set(pos, filter.evaluate(static_cast<vec2>(pos) / vec2{ width,width }));
}

void SampleWeightLUT::cookTable(const FilterWrapper& filter) {
    BuiltinRenderTarget<float> rt(*mArray);
    const auto size = mArray->size().x;
    auto buffer = std::make_unique<CommandBuffer>();
    buffer->launchKernelLinear(cookKernel, size*size, rt.toRef(), filter, size);
    Environment::get().submit(std::move(buffer)).sync();
}

SampleWeightLUT::SampleWeightLUT(unsigned tableSize, const FilterWrapper& filter) :
    mArray(std::make_shared<BuiltinArray<float>>(vec2{tableSize, tableSize})),
    mSampler(std::make_shared<BuiltinSampler<float>>(mArray->get(), cudaAddressModeMirror)) {
    cookTable(filter);
}

BuiltinSamplerRef<float> SampleWeightLUT::toRef() const {
    return mSampler->toRef();
}

DEVICE FilmTileRef::FilmTileRef(Spectrum* pixel, float* weightSum, const uvec2 size,
    const BuiltinSamplerRef<float>& weightLUT) : mPixel(pixel), mWeightSum(weightSum),
    mWidth(size.x + 2), mWeightLUT(weightLUT) {}

DEVICE void FilmTileRef::add(const vec2 pos, const Spectrum& spectrum) {
    const vec2 p = {pos.x - 0.5f, pos.y - 0.5f};
    const int x[2] = {floor(p.x), ceil(p.x)};
    const int y[2] = {floor(p.y), ceil(p.y)};
#pragma unroll
    for (auto px : x)
#pragma unroll
        for (auto py : y) {
            const auto weight = mWeightLUT.get({px - pos.x, py - pos.y});
            const auto id = (py + 1) * mWidth + (px + 1);
            mPixel[id].atomicAdd(spectrum * weight);
            atomicAdd(&mWeightSum[id], weight);
        }
}

FilmTile::FilmTile(CommandBuffer& buffer, const uvec2 size, const SampleWeightLUT& weightLUT)
    : mPixel(buffer.allocBuffer<Spectrum>((size.x + 2) * (size.y + 2))),
    mWeight(buffer.allocBuffer<float>((size.x + 2) * (size.y + 2))),
    mWeightLUT(weightLUT), mSize(size), mBuffer(buffer) {
    buffer.memset(mPixel);
    buffer.memset(mWeight);
}

GLOBAL void mergeTile(READONLY(Spectrum) ip, READONLY(float) iw, Spectrum* op, float* ow,
    const unsigned int widthSrc, const unsigned int widthDst, const uvec2 offset, const uvec2 size) {
    const ivec2 p = { offset.x + threadIdx.x - 1, offset.y + threadIdx.y - 1 };
    if (0 <= p.x & p.x < size.x & 0 <= p.y & p.y <= size.y) {
        const auto idSrc = threadIdx.y * widthSrc + threadIdx.x;
        const auto idDst = p.y * widthDst + p.x;
        op[idDst].atomicAdd(ip[idSrc]);
        atomicAdd(&ow[idDst], iw[idSrc]);
    }
}

void FilmTile::merge(const MemorySpan<Spectrum>& pixel, const MemorySpan<float>& weight,
    const uvec2 dstSize, const uvec2 offset) const {
    mBuffer.launchKernelDim(mergeTile, {}, {mSize.x + 2, mSize.y + 2}, mPixel, mWeight,
        mBuffer.useAllocated(pixel), mBuffer.useAllocated(weight), mSize.x, dstSize.x,
        offset, dstSize);
}

uvec2 FilmTile::size() const {
    return mSize;
}
