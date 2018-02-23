#include <ScanLineRenderer/SphereRasterizer.hpp>
#include <Base/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>

CUDAINLINE vec4 calcSphereRange(const vec4 sphere, const float near, const float far) {
    const auto r2 = sphere.w * sphere.w;
    constexpr auto fminv = 1e15f;
    constexpr auto fmaxv = -1e15f;
    vec4 res = {fmaxv, fminv, fmaxv, fminv};
    const auto f = [=, &res](const float posz) {
        const auto pz = fmin(fmax(sphere.z - sphere.w, fmin(fmax(posz, near), far)), sphere.z + sphere.w);
        const auto invpz = -1.0f / (epsilon<float>() + pz);
        const auto delta = pz - sphere.z;
        const auto val = fabs(sqrt(r2 - delta * delta) * invpz);
        const auto offx = sphere.x * invpz, offy = sphere.y * invpz;
        res.x = fmin(res.x, offx - val);
        res.y = fmax(res.y, offx + val);
        res.z = fmin(res.z, offy - val);
        res.w = fmax(res.w, offy + val);
    };
    f(sphere.z - sphere.w);
    f(sphere.z + sphere.w);
    constexpr auto a = 2.0f;
    const auto b = 3.0f * sphere.z;
    const auto c = r2 - sphere.z * sphere.z;
    const auto delta = b * b - 4.0f * a * c;
    if (delta >= 0.0f) {
        constexpr auto inv2a = 0.5f / a;
        const auto sqrtDelta = sqrt(delta);
        f((sqrtDelta - b) * inv2a);
        f((-sqrtDelta - b) * inv2a);
    }
    return res;
}

GLOBAL void processSphereInfoGPU(const unsigned int size,READONLY(vec4) in, SphereInfo* info,
                                 TileRef* ref, unsigned int* cnt, const vec2 fsiz, const vec2 hsiz,
                                 const float near, const float far, const vec2 mul) {
    const auto id = getID();
    if (id >= size)return;
    const auto sphere = in[id];
    if (- sphere.w > far + sphere.z | sphere.w < near + sphere.z)return;
    const auto range = calcSphereRange(sphere, near, far);
    const vec4 rect = {
        fmax(0.5f, (1.0f + range.x * mul.x) * hsiz.x - tileOffset),
        fmin(fsiz.x, (1.0f + range.y * mul.x) * hsiz.x + tileOffset),
        fmax(0.5f, (1.0f - range.w * mul.y) * hsiz.y - tileOffset),
        fmin(fsiz.y, (1.0f - range.z * mul.y) * hsiz.y + tileOffset)
    };
    if (rect.x < rect.y & rect.z < rect.w) {
        const auto tsiz = calcTileSize(rect);
        atomicInc(cnt + tsiz, maxv);
        const auto wpos = atomicInc(cnt + 6, maxv);
        ref[wpos].id = wpos;
        ref[wpos].size = tsiz;
        ref[wpos].rect = rect;
        info[wpos].id = id;
        info[wpos].info = sphere;
        info[wpos].c = length2(vec3{sphere}) - sphere.w * sphere.w;
    }
}

SphereProcessResult processSphereInfo(CommandBuffer& buffer, const MemoryRef<vec4>& spheres,
                                      const vec2 fsiz, const vec2 hsiz, const float near, const float far,
                                      const vec2 mul) {
    auto cnt = buffer.allocBuffer<unsigned int>(7);
    buffer.memset(cnt);
    auto info = buffer.allocBuffer<SphereInfo>(spheres.size());
    auto ref = buffer.allocBuffer<TileRef>(spheres.size());
    buffer.runKernelLinear(processSphereInfoGPU, spheres.size(), spheres, info, ref, cnt,
                           fsiz, hsiz, near, far, mul);
    auto sortedSphere = sortTiles(buffer, cnt, ref, spheres.size() * 2U + 2048U);
    cnt.earlyRelease();
    ref.earlyRelease();
    return SphereProcessResult(sortedSphere.first, info, sortedSphere.second);
}
