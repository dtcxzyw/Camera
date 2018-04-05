#include <Rasterizer/PostProcess.hpp>
#include "kernel.hpp"
#include <Rasterizer/SphereRasterizer.hpp>

CUDAINLINE vec4 vsSphere(vec4 sp, const Uniform& uniform) {
    return calcCameraSphere(sp, uniform.V);
}

CUDAINLINE void drawSpherePoint(unsigned int id, ivec2 uv, float, vec3, vec3, float, 
                                bool,vec2, const Uniform&, FrameBufferRef& fbo) {
    fbo.color.set(uv, { (id+1)/10.0f,1.0f,1.0f,1.0f });
}

CUDAINLINE void post(ivec2 NDC, const FrameBufferRef& uni, BuiltinRenderTargetRef<RGBA8> out) {
    const RGB c = uni.color.get(NDC);
    const RGBA8 color = { c * 255.0f, 255 };
    out.set(NDC, color);
}

void kernel(const MemoryRef<vec4>& spheres,
            const MemoryRef<Uniform>& uniform, FrameBuffer& fbo,
            const Camera::RasterPosConverter converter, CommandBuffer& buffer) {
    //fbo.colorRT->clear(buffer, {});
    const auto frameBuffer = fbo.getData(buffer);
    const vec4 scissor = { 0.0f,fbo.size.x,0.0f,fbo.size.y };
    renderSpheres<Uniform,FrameBufferRef,vsSphere,drawSpherePoint>(buffer,
        spheres, uniform, frameBuffer, fbo.size, converter.near, converter.far, converter.mul, scissor);
    renderFullScreen<FrameBufferRef, BuiltinRenderTargetRef<RGBA8>, post>(buffer, frameBuffer,
        fbo.postRT->toTarget(), fbo.size);
}
