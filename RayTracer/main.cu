#include "kernel.hpp"
#include <cstdio>
#include <IO/Image.hpp>
#include <thread>
using namespace std::chrono_literals;

Camera camera;
DisneyBRDFArg arg;

void setUIStyle() {
    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.Alpha = 0.8f;
    style.AntiAliasedFill = true;
    style.AntiAliasedLines = true;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameRounding = 5.0f;
}

void renderGUI(IMGUIWindow& window) {
    window.newFrame();
    ImGui::Begin("Debug");
    ImGui::SetWindowPos({ 0, 0 });
    ImGui::SetWindowSize({ 500,550 });
    ImGui::SetWindowFontScale(1.5f);
    ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
    ImGui::Text("FOV %.1f ",degrees(camera.toFov()));
    ImGui::SliderFloat("focal length",&camera.focalLength,1.0f,500.0f,"%.1f");

#define COLOR(name)\
arg.##name=clamp(arg.##name,vec3(0.01f),vec3(0.999f));\
ImGui::ColorEdit3(#name,&arg.##name[0],ImGuiColorEditFlags_Float);\

    COLOR(baseColor);
    //Color(edgeTint);
#undef COLOR

#define ARG(name)\
 arg.##name=clamp(arg.##name,0.01f,0.999f);\
 ImGui::SliderFloat(#name, &arg.##name, 0.01f, 0.999f);\

    ARG(metallic);
    ARG(subsurface);
    ARG(specular);
    ARG(roughness);
    ARG(specularTint);
    ARG(anisotropic);
    ARG(sheen);
    ARG(sheenTint);
    ARG(clearcoat);
    ARG(clearcoatGloss);
#undef ARG
    ImGui::End();
    window.renderGUI();
}

using SwapChainT = SwapChain<FrameBufferCPU>;
struct RenderingTask {
    Future future;
    SwapChainT::SharedFrame frame;
    RenderingTask(const Future& fut, const SwapChainT::SharedFrame& fbo)
    :future(fut), frame(fbo){}
};

int main() {
    getEnvironment().init();
    try {
        camera.near = 1.0f;
        camera.far = 200.0f;
        camera.filmAperture = { 0.980f,0.735f };
        camera.mode = Camera::FitResolutionGate::Overscan;
        camera.focalLength = 15.0f;

        Stream resLoader;

        arg.baseColor = vec3{220,223,227}/255.0f;

        IMGUIWindow window;
        setUIStyle();
        ImGui::GetIO().WantCaptureKeyboard = true;

        SwapChainT swapChain(3);
        std::queue<RenderingTask> tasks;
        {
            DispatchSystem system(2);
            auto lum = allocBuffer<float>();
            while (window.update()) {
                const auto size = window.size();
                if (size.x == 0 || size.y == 0) {
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                SwapChainT::SharedFrame frame;
                while (true) {
                    system.update(1ms);
                    if (!swapChain.empty())
                        tasks.push(addTask(system, swapChain.pop(), size, lum.begin()));
                    if (!tasks.empty() && tasks.front().future.finished()) {
                        frame = tasks.front().frame;
                        tasks.pop();
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
    catch (const std::runtime_error& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}

