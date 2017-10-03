#include <ScanLineRenderer/ScanLineRenderer.hpp>
#include "kernel.hpp"
#include <fstream>
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <chrono>
#include <sm_20_intrinsics.h>

CUDA void VS(VI in, Uniform uniform, OI& out, vec4& pos) {
    pos = uniform.mat* vec4(in.pos, 1.0f);
    out.get<coord>() = in.uv;
}

CUDA void drawPoint(ivec2 uv, float z,OI out, Uniform uniform, FrameBufferGPU& fbo) {
    uv.y = fbo.mSize.y - uv.y;
    if (fbo.depth.get(uv) > z) {
        auto color = uniform.tex.get(out.get<coord>());
        for (int i = 0; i < 64; ++i) {
            auto fz = fbo.depth.get(uv);
            if (fz > z | (fz==z & fbo.color.get(uv) != color)) {
                fbo.depth.set(uv, z);
                __threadfence_system();
                fbo.color.set(uv, color);
                __threadfence_system();
            }
        }
    }
}

inline int toInt(float x) { return clamp(static_cast<int>(clamp(x,0.0f,1.0f) * 255.0f + 0.5f),0,255); }

void toPNG(RGBA* p,ivec2 size) {
    std::vector<unsigned char> rgba(size.x*size.y * 4);
    unsigned char *ptr = rgba.data();
    for (int i = 0; i < size.x*size.y; ++i) {
        RGBA c = p[i];
        *ptr++ = toInt(c.r);
        *ptr++ = toInt(c.g);
        *ptr++ = toInt(c.b);
        *ptr++ = toInt(c.a);
    }
    stbi_write_png("output.png", size.x, size.y, STBI_rgb_alpha, rgba.data(),0);
}

void kernel(DataViewer<VI> vbo, DataViewer<uvec3> ibo,DataViewer<Uniform> uniform,
    FrameBufferCPU& fbo, Pipeline& pipeline) {
    using Clock = std::chrono::high_resolution_clock;
    fbo.colorBuffer->clear(pipeline,vec4{ 0.0f,0.0f,0.0f,1.0f });
    fbo.depthBuffer->clear(pipeline,1.0f);
    auto b = Clock::now();
    renderTriangles<VI, OI, Uniform, FrameBufferGPU, VS, drawPoint>
        (pipeline,vbo, ibo, uniform, fbo.dataGPU,fbo.size);
    printf("%.3lf ms\n",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - b).count() / 1000.0);
    //auto data = fbo.colorBuffer->download(pipeline);
    //pipeline.sync();
    //toPNG(data.begin(), fbo.size);
}

