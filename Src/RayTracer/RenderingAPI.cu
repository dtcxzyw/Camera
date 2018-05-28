#include <RayTracer/RenderingAPI.hpp>
#include <RayTracer/Integrator.hpp>
#include <RayTracer/Film.hpp>
#include <Core/Environment.hpp>

GLOBAL void divWeight(const unsigned int size, Spectrum* pixel, READONLY(float) weight) {
    const auto id = getId();
    if (id >= size)return;
    const auto w = weight[id];
    auto&& col = pixel[id];
    col = w > 0.0f ? col / w : Spectrum{};
}

constexpr auto tileSize = 30U;

MemorySpan<Spectrum> renderFrame(Integrator& integrator,
    const SceneDesc& scene, const Transform& cameraTransform,
    const RayGeneratorWrapper& rayGenerator, const SampleWeightLUT& weightLUT,
    const uvec2 size) {
    MemorySpan<Spectrum> pixel(size.x * size.y);
    pixel.memset();
    MemorySpan<float> weight(size.x * size.y);
    weight.memset();
    std::vector<Future> tasks;
    const auto sx = calcBlockSize(size.x, tileSize);
    const auto sy = calcBlockSize(size.y, tileSize);
    for (auto x = 0; x < sx; ++x)
        for (auto y = 0; y < sy; ++y) {
            const uvec2 lt = {x * tileSize, y * tileSize};
            const auto rb = min(size, uvec2{(x + 1) * tileSize, (y + 1) * tileSize});
            const auto currentTileSize = rb - lt;
            auto buffer = std::make_unique<CommandBuffer>();
            {
                const auto tile = std::make_unique<FilmTile>(*buffer, currentTileSize, weightLUT);
                integrator.render(*buffer, scene, cameraTransform, rayGenerator, *tile, lt);
                tile->merge(pixel, weight, size, lt);
            }
            tasks.emplace_back(Environment::get().submit(std::move(buffer)));
        }
    for (auto&& task : tasks)
        task.sync();
    {
        auto buffer = std::make_unique<CommandBuffer>();
        buffer->launchKernelLinear(makeKernelDesc(divWeight), pixel.size(), buffer->useAllocated(pixel),
            buffer->useAllocated(weight));
        Environment::get().submit(std::move(buffer)).sync();
    }
    return pixel;
}
