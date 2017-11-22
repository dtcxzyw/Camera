#include <cstdio>
#include <system_error>
#include <Interaction/OpenGL.hpp>
#include "kernel.hpp"
#include <PBR/PhotorealisticRendering.hpp>
#include <thread>
using namespace std::chrono_literals;

auto f = 50.0f, roughness = 0.5f,light=5.0f,metallic=0.01f,ao=0.05f;
StaticMesh model;

void renderGUI(IMGUIWindow& window) {
    window.newFrame();
    ImGui::Begin("Debug");
    ImGui::SetWindowPos({ 0, 0 });
    ImGui::SetWindowSize({ 500,300 });
    ImGui::SetWindowFontScale(1.5f);
    ImGui::StyleColorsDark();
    ImGui::Text("vertices %d, triangles: %d\n", static_cast<int>(model.mVert.size()),
        static_cast<int>(model.mIndex.size()));
    ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
    ImGui::SliderFloat("focal length",&f,15.0f,150.0f,"%.1f");
    ImGui::SliderFloat("roughness", &roughness, 0.01f, 0.99f);
    ImGui::SliderFloat("metallic", &metallic, 0.01f, 0.99f);
    ImGui::SliderFloat("ao", &ao, 0.001f, 0.09f);
    ImGui::SliderFloat("light", &light, 0.0f, 10.0f, "%.1f");
    ImGui::End();
    window.renderGUI();
}

int main() {
    getEnvironment().init();
    try {
        model.load("Res/bunny.obj");
        FrameBufferCPU FB;
        IMGUIWindow window;
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
            auto fov = toFOV(36.0f*24.0f,f);
            glm::mat4 P = perspectiveFov(fov, w, h, 1.0f, 20.0f);
            float now = glfwGetTime();
            float delta = now - t;
            M = rotate(M, delta*0.2f, { 0.0f,1.0f,0.0f });
            t = now;
            Uniform u;
            u.VP = P*V;
            u.M = M;
            u.invM = mat3(transpose(inverse(M)));
            u.lc = vec3(light);
            u.albedo = {1.000f, 0.766f, 0.336f};
            u.cp = cp;
            u.dir = normalize(lp);
            u.roughness = roughness;
            u.metallic = metallic;
            u.ao = ao;
            uniform.set(u, pipeline);
            BuiltinRenderTarget<RGBA> RT(window.map(pipeline,size),size);
            auto sum = allocBuffer<std::pair<float,unsigned int>>();
            sum->first = 0.0f;
            sum->second = 0;
            PostUniform post;
            post.in = FB.data;
            auto tw = powf(0.2f, delta);
            auto nlum = lum*(1.0f - tw) + last*tw;
            last = nlum;
            post.lum = fmax(calcLum(nlum),0.1f);
            post.sum = sum.begin();
            puni.set(post, pipeline);
            kernel(model.mVert, model.mIndex, uniform.get(), FB, puni.get(),RT.toTarget(), pipeline);
            window.unmapAndPresent(pipeline);
            renderGUI(window);
            window.swapBuffers();
            pipeline.sync();
            lum =sum->first/sum->second;
        }
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}
