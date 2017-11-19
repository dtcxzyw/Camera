#include <cstdio>
#include <system_error>
#include "kernel.hpp"
#include <Interaction/OpenGL.hpp>
#include <PBR/PhotorealisticRendering.hpp>
#include <thread>
using namespace std::chrono_literals;

int main() {
    getEnvironment().init();
    try {
        StaticMesh model;
        model.load("Res/bunny.obj");
        printf("vertices %d ,triangles: %d\n", static_cast<int>(model.mVert.size()),
            static_cast<int>(model.mIndex.size()));
        FrameBufferCPU FB;
        GLWindow window;
        Pipeline pipeline;
        vec3 cp = { 10.0f,0.0f,0.0f },lp = { 10.0f,4.0f,0.0f };
        auto V = lookAt(cp, { 0.0f,0.0f,0.0f }, { 0.0f,1.0f,0.0f });
        glm::mat4 M;
        M = scale(M, vec3(1.0f, 1.0f, 1.0f)*10.0f);
        float t = glfwGetTime(),lum=1.0f,last=1.0f;
        Constant<Uniform> uniform;
        Constant<PostUniform> puni;
        while (window.update()) {
            auto size = window.size();
            if (size.x == 0 || size.y == 0) {
                std::this_thread::sleep_for(1ms);
                continue;
            }
            FB.resize(size.x, size.y, pipeline);
            float w = size.x, h = size.y;
            auto fov = toFOV(36.0f*24.0f,50.0f);
            glm::mat4 P = perspectiveFov(fov, w, h, 1.0f, 20.0f);
            float now = glfwGetTime();
            float delta = now - t;
            M = rotate(M, delta*0.2f, { 0.0f,1.0f,0.0f });
            printf("\r%f %.2f ms          ",degrees(fov),delta*1000.0f);
            t = now;
            Uniform u;
            u.VP = P*V;
            u.M = M;
            u.invM = mat3(transpose(inverse(M)));
            u.lc = vec3(5.0f);
            u.color = {1.000f, 0.766f, 0.336f};
            u.cp = cp;
            u.dir = normalize(lp);
            u.roughness = 0.5f;
            u.f0 = { 1.00f, 0.71f, 0.29f };
            //u.off = (sin(now) + 1.0f)*0.01f;
            u.off = 0.0f;
            uniform.set(u, pipeline);
            BuiltinRenderTarget<RGBA> RT(window.map(pipeline,size),size);
            auto sum = allocBuffer<float>();
            *sum = 0.0f;
            PostUniform post;
            post.in = FB.data;
            auto tw = powf(0.2f, delta);
            auto nlum = lum*(1.0f - tw) + last*tw;
            last = nlum;
            post.lum =calcLum(nlum);
            post.sum = sum.begin();
            puni.set(post, pipeline);
            kernel(model.mVert, model.mIndex, uniform.get(), FB, puni.get(),RT.toTarget(), pipeline);
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
