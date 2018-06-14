#include <RayTracer/Film.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>
#include <Core/DeviceFunctions.hpp>

GLOBAL void cookKernel(const uint32_t size, BuiltinRenderTargetRef<float> dst,
    FilterWrapper filter, const uint32_t width) {
    const auto id = getId();
    if (id >= size)return;
    const ivec2 pos = {id / width, id % width};
    dst.set(pos, filter.evaluate(static_cast<vec2>(pos) / vec2{width, width}));
}

void SampleWeightLUT::cookTable(const FilterWrapper& filter) {
    BuiltinRenderTarget<float> rt(*mArray);
    const auto size = mArray->size().x;
    auto buffer = std::make_unique<CommandBuffer>();
    buffer->launchKernelLinear(makeKernelDesc(cookKernel), size * size, rt.toRef(), filter, size);
    Environment::get().submit(std::move(buffer)).sync();
}

SampleWeightLUT::SampleWeightLUT(const uint32_t tableSize, const FilterWrapper& filter) :
    mArray(std::make_shared<BuiltinArray<float>>(vec2{tableSize, tableSize})),
    mSampler(std::make_shared<BuiltinSampler<float>>(mArray->get(), cudaAddressModeMirror)) {
    cookTable(filter);
}

BuiltinSamplerRef<float> SampleWeightLUT::toRef() const {
    return mSampler->toRef();
}

FilmTileRef::FilmTileRef(Spectrum* pixel, float* weightSum, const uvec2 size,
    const BuiltinSamplerRef<float>& weightLUT) : mPixel(pixel), mWeightSum(weightSum),
    mWidth(size.x + 2), mWeightLUT(weightLUT) {}

DEVICE void FilmTileRef::add(const vec2 pos, const Spectrum& spectrum) {
    const int bx = floor(pos.x - 0.5f), by = floor(pos.y - 0.5f);
    for (auto ox = 0; ox < 2; ++ox)
        for (auto oy = 0; oy < 2; ++oy) {
            const auto cx = bx + ox, cy = by + oy;
            const auto weight = mWeightLUT.get({cx + 0.5f - pos.x, cy + 0.5f - pos.y});
            const auto id = (cy + 1) * mWidth + (cx + 1);
            mPixel[id].atomicAdd(spectrum * weight);
            deviceAtomicAdd(&mWeightSum[id], weight);
        }
}

FilmTile::FilmTile(CommandBuffer& buffer, const uvec2 size, const SampleWeightLUT& weightLUT)
    : mPixel(buffer.allocBuffer<Spectrum>((size.x + 2) * (size.y + 2))),
    mWeight(buffer.allocBuffer<float>((size.x + 2) * (size.y + 2))),
    mWeightLUT(weightLUT), mSize(size), mBuffer(buffer) {
    buffer.memset(mPixel);
    buffer.memset(mWeight);
}

GLOBAL void mergeTile(const uint32_t size, READONLY(Spectrum) ip, READONLY(float) iw,
    Spectrum* op, float* ow, const uint32_t widthSrc, const uint32_t widthDst,
    const uvec2 offset, const uvec2 dstSize) {
    const auto id = getId();
    if (id >= size)return;
    const auto px = id % widthSrc, py = id / widthSrc;
    const uvec2 p = {offset.x + px - 1, offset.y + py - 1};
    if (p.x < dstSize.x & p.y < dstSize.y) {
        const auto idSrc = py * widthSrc + px;
        const auto idDst = p.y * widthDst + p.x;
        op[idDst].atomicAdd(ip[idSrc]);
        deviceAtomicAdd(&ow[idDst], iw[idSrc]);
    }
}

void FilmTile::merge(const MemorySpan<Spectrum>& pixel, const MemorySpan<float>& weight,
    const uvec2 dstSize, const uvec2 offset) const {
    const auto siz = (mSize.x + 2) * (mSize.y + 2);
    mBuffer.launchKernelLinear(makeKernelDesc(mergeTile), siz, mPixel, mWeight,
        mBuffer.useAllocated(pixel), mBuffer.useAllocated(weight), mSize.x + 2, dstSize.x,
        offset, dstSize);
}

uvec2 FilmTile::size() const {
    return mSize;
}
