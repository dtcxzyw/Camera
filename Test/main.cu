#include <stdio.h>
#include <system_error>
#include "kernel.hpp"
#include <Interaction/OpenGL.hpp>
#include <thread>
using namespace std::chrono_literals;

int main() {
    getEnvironment().init();
    try {
        StaticMesh model;
        model.load("Res/bunny.ply");
        printf("vertices %d ,triangles: %d\n", static_cast<int>(model.mVert.size()),
            static_cast<int>(model.mIndex.size()));
        FrameBufferCPU FB;
        GLWindow window;
        Pipeline pipeline;
        glm::mat4 V = lookAt({ 10.0f,4.0f,0.0f }, vec3{ 0.0f,4.0f,0.0f }, { 0.0f,1.0f,0.0f });
        glm::mat4 M;
        M = scale(M, vec3(1.0f, 1.0f, 1.0f)*40.0f);
        float t = glfwGetTime(),lum=1.0f,last=1.0f;
        while (window.update()) {
            pipeline.sync();
            auto size = window.size();
            if (size.x == 0 || size.y == 0) {
                std::this_thread::sleep_for(1ms);
                continue;
            }
            FB.resize(size.x, size.y);
            float w = size.x, h = size.y;
            glm::mat4 P = perspectiveFov(radians(45.0f), w, h, 1.0f, 20.0f);
            float now = glfwGetTime();
            float delta = now - t;
            M = rotate(M, delta*0.2f, { 0.0f,1.0f,0.0f });
            printf("\r%.2f ms          ", delta*1000.0f);
            t = now;
            auto uniform = allocBuffer<Uniform>();
            Uniform u;
            u.VP = P*V;
            u.M = M;
            u.invM = mat3(transpose(inverse(M)));
            u.lc = vec3(5.0f);
            u.color = {1.0f,0.84f,0.0f};
            u.cp = { 10.0f,4.0f,0.0f };
            u.dir = normalize(u.cp);
            u.roughness = 0.5f;
            u.f0 = { 1.00f, 0.71f, 0.29f };
            uniform[0] = u;
            BuiltinRenderTarget<RGBA> RT(window.map(pipeline,size),size);
            auto post = allocBuffer<PostUniform>();
            auto sum = allocBuffer<float>();
            *sum = 0.0f;
            post->in = *FB.dataGPU;
            auto tw = powf(0.2f, delta);
            auto nlum = lum*(1.0f - tw) + last*tw;
            last = nlum;
            post->lum =calcLum(nlum);
            post->sum = sum.begin();
            kernel(model.mVert, model.mIndex, uniform, FB, post,RT.toTarget(), pipeline);
            window.unmapAndPresent(pipeline);
            pipeline.sync();
            lum =*sum/(w*h);
        }
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}
