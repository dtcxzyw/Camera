#include "kernel.hpp"
#include <cstdio>
#include <IO/Image.hpp>
#include <thread>
#include <Base/Environment.hpp>
#include <Interaction/SwapChain.hpp>
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <Base/CompileEnd.hpp>

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

void renderGUI(D3D11Window& window) {
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
}

using SwapChainT = SwapChain<FrameBufferCPU>;
struct RenderingTask {
    Future future;
    SwapChainT::SharedFrame frame;
    RenderingTask(const Future& fut, const SwapChainT::SharedFrame& fbo)
    :future(fut), frame(fbo){}
};

int main() {
    auto&& window = D3D11Window::get();
    setUIStyle();
    ImGui::GetIO().WantCaptureKeyboard = true;

    auto&& env = Environment::get();
    env.init(AppType::Offline,GraphicsInteroperability::D3D11);

    try {
        camera.near = 1.0f;
        camera.far = 200.0f;
        camera.filmAperture = { 0.980f,0.735f };
        camera.mode = Camera::FitResolutionGate::Overscan;
        camera.focalLength = 15.0f;

        Stream resLoader;

        arg.baseColor = vec3{220,223,227}/255.0f;

        SwapChainT swapChain(3);
        std::queue<RenderingTask> tasks;
        {
            Stream copyStream;
            window.bindBackBuffer(copyStream.get());
            auto lum = DataViewer<float>();
            while (window.update()) {
                const auto size = window.size();
                if (size.x == 0 || size.y == 0) {
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                SwapChainT::SharedFrame frame;
                while (true) {
                    if (!tasks.empty() && tasks.front().future.finished()) {
                        frame = tasks.front().frame;
                        tasks.pop();
                        break;
                    }
                }
                if (frame->size == size) {
                    window.present(frame->postRT->get());
                    renderGUI(window);
                    window.swapBuffers();
                }
                swapChain.push(std::move(frame));
            }
            window.unbindBackBuffer();
        }
        env.uninit();
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}

