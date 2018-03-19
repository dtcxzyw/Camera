#include <ScanLineRenderer/SphereRasterizer.hpp>
#include <Base/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>

CUDAINLINE bool cmpMin(const float a, const float b) {
    return a < b;
}

CUDAINLINE bool cmpMax(const float a, const float b) {
    return a > b;
}

template <typename Func,typename Cmp>
CUDAINLINE float calcValue(const Func& func, const Cmp& cmp, float l, float r) {
    constexpr auto eps = 1e-3f;
    while (r - l >= eps) {
        const auto lm = (l * 2.0f + r) / 3.0f;
        const auto rm = (l + r * 2.0f) / 3.0f;
        if (cmp(func(lm), func(rm)))r = rm;
        else l = lm;
    }
    return func(l);
}

CUDAINLINE bool calcSphereRange(const vec4 sphere, const float near, const float far,vec4& res) {
    const auto r2 = sphere.w * sphere.w;
    const auto begin = fmax(-far, sphere.z - sphere.w), end = fmin(-near, sphere.z + sphere.w);
    if (begin >= end)return false;
    res.x = calcValue([=](const float z) {
        const auto dz = z - sphere.z;
        return (sphere.x - sqrt(r2 - dz * dz)) / -z;
    }, cmpMin, begin, end);
    res.y = calcValue([=](const float z) {
        const auto dz = z - sphere.z;
        return (sphere.x + sqrt(r2 - dz * dz)) / -z;
    }, cmpMax, begin, end);
    res.z = calcValue([=](const float z) {
        const auto dz = z - sphere.z;
        return (sphere.y - sqrt(r2 - dz * dz)) / -z;
    }, cmpMin, begin, end);
    res.w = calcValue([=](const float z) {
        const auto dz = z - sphere.z;
        return (sphere.y + sqrt(r2 - dz * dz)) / -z;
    }, cmpMax, begin, end);
    return true;
}

GLOBAL void processSphereInfoGPU(const unsigned int size,READONLY(vec4) in, SphereInfo* info,
                                 TileRef* ref, unsigned int* cnt, const vec4 scissor, const vec2 hsiz,
                                 const float near, const float far, const vec2 mul) {
    const auto id = getId();
    if (id >= size)return;
    const auto sphere = in[id];
    vec4 range;
    if (!calcSphereRange(sphere, near, far, range))return;
    const uvec4 rect = {
        fmax(scissor.x, (1.0f + range.x * mul.x) * hsiz.x - tileOffset),
        fmin(scissor.y, (1.0f + range.y * mul.x) * hsiz.x + tileOffset),
        fmax(scissor.z, (1.0f - range.w * mul.y) * hsiz.y - tileOffset),
        fmin(scissor.w, (1.0f - range.z * mul.y) * hsiz.y + tileOffset)
    };
    if (rect.x < rect.y & rect.z < rect.w) {
        const auto tsiz = calcTileSize(rect);
        atomicInc(cnt + tsiz, maxv);
        const auto wpos = atomicInc(cnt + 6, maxv);
        ref[wpos].id = wpos;
        ref[wpos].size = tsiz;
        ref[wpos].rect = rect;
        info[wpos].id = id;
        info[wpos].info = {vec3{sphere}, 1.0f / sphere.w};
        info[wpos].c = length2(vec3{sphere}) - sphere.w * sphere.w;
    }
}

SphereProcessResult processSphereInfo(CommandBuffer& buffer, const MemoryRef<vec4>& spheres,
                                      const vec4 scissor, const vec2 hsiz, const float near, const float far,
                                      const vec2 mul) {
    auto cnt = buffer.allocBuffer<unsigned int>(7);
    buffer.memset(cnt);
    auto info = buffer.allocBuffer<SphereInfo>(spheres.size());
    auto ref = buffer.allocBuffer<TileRef>(spheres.size());
    buffer.runKernelLinear(processSphereInfoGPU, spheres.size(), spheres, info, ref, cnt,
                           scissor, hsiz, near, far, mul);
    auto sortedSphere = sortTiles(buffer, cnt, ref, spheres.size() * 2U + 2048U, spheres.size());
    cnt.earlyRelease();
    ref.earlyRelease();
    return SphereProcessResult(sortedSphere.first, info, sortedSphere.second);
}
