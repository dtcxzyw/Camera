#include <RayTracer/Film.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Environment.hpp>

GLOBAL void cookKernel(BuiltinRenderTargetRef<float> dst, FilterWrapper filter) {
    const ivec2 id = { threadIdx.x,threadIdx.y };
    dst.set(id, filter.evaluate(vec2{ id.x / blockDim.x,id.y / blockDim.y }));
}

void SampleWeightLUT::cookTable(const FilterWrapper& filter) {
    BuiltinRenderTarget<float> rt(*mArray);
    const auto size = mArray->size();
    auto buffer = std::make_unique<CommandBuffer>();
    buffer->launchKernelDim(cookKernel, {}, dim3{ size.x,size.y,1 }, rt.toRef(),filter);
    Environment::get().submit(std::move(buffer)).sync();
}

DEVICE void FilmTileRef::add(const vec2 pos, const Spectrum& spectrum) {
    NOT_IMPLEMENTED();
}

void FilmTile::download(CommandBuffer& buffer, PinnedBuffer<Spectrum>& dst,
    uvec2 dstSize, uvec2 offset) const {
    NOT_IMPLEMENTED();
}
