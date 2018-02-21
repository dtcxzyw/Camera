#include <ScanLineRenderer/SphereRasterizer.hpp>
#include <Base/CompileBegin.hpp>
#include <device_atomic_functions.h>
#include <Base/CompileEnd.hpp>

GLOBAL void processSphereInfoGPU(const unsigned int size,READONLY(vec4) in, SphereInfo* info,
                                 TileRef* ref, unsigned int* cnt, const vec2 fsiz, const vec2 hsiz,
                                 const float near, const float far, const vec2 mul) {
    const auto id = getID();
    if (id >= size)return;
    const auto sphere = in[id];
    vec3 vpos = {sphere.x * mul.x, sphere.y * mul.y, -sphere.z};
    if (vpos.z - sphere.w > far | vpos.z + sphere.w < near)return;
    vpos = toRaster(vpos, hsiz);
    const auto r = sphere.w * vpos.z * mul;
    const vec4 rect = {
        fmax(0.5f, vpos.x - r.x-tileOffset),
        fmin(fsiz.x, vpos.x + r.x+tileOffset),
        fmax(0.5f, vpos.y - r.y-tileOffset),
        fmin(fsiz.y, vpos.y + r.y+tileOffset)
    };
    if(rect.x<rect.y & rect.z<rect.w) {
        const auto tsiz = calcTileSize(rect);
        atomicInc(cnt+tsiz,maxv);
        const auto wpos = atomicInc(cnt + 6, maxv);
        ref[wpos].id = wpos;
        ref[wpos].size = size;
        ref[wpos].rect = rect;
        info[wpos].id = id;
        info[wpos].info = sphere;
        info[wpos].c = length2(vec3{ sphere })-sphere.w*sphere.w;
    }
}

SphereProcessResult processSphereInfo(CommandBuffer& buffer, const DataPtr<vec4>& spheres,
                                      const vec2 fsiz, const vec2 hsiz, const float near, const float far,
                                      const vec2 mul) {
    auto cnt = buffer.allocBuffer<unsigned int>(7);
    buffer.memset(cnt);
    auto info = buffer.allocBuffer<SphereInfo>(spheres.size());
    auto ref = buffer.allocBuffer<TileRef>(spheres.size());
    buffer.runKernelLinear(processSphereInfoGPU, spheres.size(), spheres.get(), info, ref, cnt,
                           fsiz, hsiz, near, far, mul);
    auto sortedSphere = sortTiles(buffer,cnt,ref);
    cnt.earlyRelease();
    ref.earlyRelease();
    return SphereProcessResult(sortedSphere.first,info,sortedSphere.second);
}
