#include "kernel.hpp"
#include <cstdio>
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

Uniform getUniform(float w,float h,float delta) {
    vec3 cp = { 10.0f,0.0f,0.0f }, lp = { 10.0f,4.0f,0.0f };
    auto V = lookAt(cp, { 0.0f,0.0f,0.0f }, { 0.0f,1.0f,0.0f });
    static glm::mat4 M = scale(glm::mat4{}, vec3(1.0f, 1.0f, 1.0f)*10.0f);
    auto fov = toFOV(36.0f*24.0f, f);
    glm::mat4 P = perspectiveFov(fov, w, h, 1.0f, 20.0f);
    M = rotate(M, delta*0.2f, { 0.0f,1.0f,0.0f });
    Uniform u;
    u.VP = P * V;
    u.M = M;
    u.invM = mat3(transpose(inverse(M)));
    u.lc = vec3(light);
    u.albedo = { 1.000f, 0.766f, 0.336f };
    u.cp = cp;
    u.dir = normalize(lp);
    u.roughness = roughness;
    u.metallic = metallic;
    u.ao = ao;
    return u;
}

using SwapChain_t = SwapChain<FrameBufferCPU>;

auto addTask(DispatchSystem& system
    ,SwapChain_t::SharedFrame frame,uvec2 size,float* lum) {
    static float last = glfwGetTime();
    float now = glfwGetTime();
    auto uniform = getUniform(size.x,size.y,now-last);
    last = now;
    auto buffer=std::make_unique<CommandBuffer>();
    frame->resize(size.x,size.y);
    auto uni = buffer->allocConstant<Uniform>();
    buffer->memcpy(uni, [uniform](auto call) {call(&uniform); });
    kernel(model.mVert,model.mIndex,uni,*frame,lum,*buffer);
    return std::make_pair(system.submit(std::move(buffer)),frame);
}

int main() {
    getEnvironment().init();
    try {
        model.load("Res/bunny.obj");
        IMGUIWindow window;
        SwapChain_t swapChain(8);
        std::queue<std::pair<Future, SwapChain_t::SharedFrame>> futures;
        {
            DispatchSystem system(3);
            auto lum = allocBuffer<float>();
            while (window.update()) {
                auto size = window.size();
                if (size.x == 0 || size.y == 0) {
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                SwapChain_t::SharedFrame frame;
                while (true) {
                    system.update(500us);
                    if (!swapChain.empty())
                        futures.emplace(addTask(system, swapChain.pop(), size, lum.begin()));
                    if (futures.size() && futures.front().first.finished()) {
                        frame = futures.front().second;
                        futures.pop();
                        break;
                    }
                }
                window.present(frame->image);
                swapChain.push(std::move(frame));
                renderGUI(window);
                window.swapBuffers();
            }
        }
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}

