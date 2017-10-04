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
        FrameBufferCPU FB;
        GLWindow window;
        Pipeline pipeline;
        glm::mat4 V = lookAt({ 10.0f,3.0f,0.0f }, vec3{ 0.0f,3.0f,0.0f }, { 0.0f,1.0f,0.0f });
        glm::mat4 M;
        M = scale(M, vec3(1.0f, 1.0f, 1.0f)*30.0f);
        float t = glfwGetTime();
        while (window.update()) {
            auto size = window.size();
            if (size.x == 0 || size.y == 0)continue;
            FB.resize(size.x, size.y);
            float w = size.x, h = size.y;
            glm::mat4 P = perspectiveFov(radians(45.0f), w, h, 1.0f, 15.0f);
            float now = glfwGetTime();
            M = rotate(M, (now - t)*0.2f, { 0.0f,1.0f,0.0f });
            printf("%.2f ms\n", (now - t)*1000.0f);
            t = now;
            Uniform uni{ P*V*M };
            auto uniform = share(std::vector<Uniform>({ uni }));
            kernel(model.mVert, model.mIndex, uniform, FB, pipeline);
            window.present(pipeline, *FB.colorBuffer);
        }
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}
