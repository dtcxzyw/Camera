#include <stdio.h>
#include <system_error>
#include "kernel.hpp"
#include <Interaction/OpenGL.hpp>

int main() {
    getDevice().init(0);
    try {
        StaticMesh model;
        model.load("Res/piano.obj");
        FrameBufferCPU FB;
        GLWindow window;
        Pipeline pipeline;
        glm::mat4 M;
        M = scale(M, vec3(1.0f, 1.0f, 1.0f)*0.03f);
        float t = glfwGetTime();
        while (window.update()) {
            auto size = window.size();
            FB.resize(size.x,size.y);
            float w = size.x, h = size.y;
            glm::mat4 P = perspectiveFov(radians(45.0f), w, h, 1.0f, 20.0f);
            glm::mat4 V = lookAt({ 6.0f,6.0f,6.0f }, vec3{ 0.0f,2.7f,0.0f }, { 0.0f,1.0f,0.0f });
            float now = glfwGetTime();
            M = rotate(M, (now-t)*0.03f, { 0.0f,1.0f,0.0f });
            t = now;
            auto uniform = share(std::vector<Uniform>({ { P*V*M } }));
            kernel(model.mVert, model.mIndex, uniform, FB,pipeline);
            window.present(pipeline, *FB.colorBuffer);
        }
    }
    catch (const std::exception& e) {
        printf("Error:%s\n", e.what());
    }
    return 0;
}
