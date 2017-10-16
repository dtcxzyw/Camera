#include <stdio.h>
#include <system_error>
#include "kernel.hpp"
#include <Interaction/OpenGL.hpp>
#include <thread>

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
        float t = glfwGetTime();
        while (window.update()) {
            pipeline.sync();
            auto size = window.size();
            if (size.x == 0 || size.y == 0)continue;
            FB.resize(size.x, size.y);
            float w = size.x, h = size.y;
            glm::mat4 P = perspectiveFov(radians(45.0f), w, h, 1.0f, 20.0f);
            float now = glfwGetTime();
            M = rotate(M, (now - t)*0.2f, { 0.0f,1.0f,0.0f });
            //M = translate(M, (now - t)*0.02f*vec3 { 1.0f, 0.0f, 0.0f });
            printf("\r%.2f ms          ", (now - t)*1000.0f);
            t = now;
            auto uniform = allocBuffer<Uniform>(1);
            uniform[0] = { P*V*M };
            BuiltinRenderTarget<RGBA> RT(window.map(pipeline,size),size);
            kernel(model.mVert, model.mIndex, uniform, FB,RT.toTarget(),pipeline);
            window.unmapAndPresent(pipeline);
        }
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}
