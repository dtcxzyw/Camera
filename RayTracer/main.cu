#include "kernel.hpp"
#include <cstdio>
#include <IO/Image.hpp>
#include <thread>
#include <Core/Environment.hpp>
#include <Interaction/SwapChain.hpp>
#include <Core/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <Core/CompileEnd.hpp>
#include <Camera/PinholeCamera.hpp>
#include <Interaction/D3D11.hpp>

using namespace std::chrono_literals;

PinholeCamera camera;

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
        camera.mode = PinholeCamera::FitResolutionGate::Overscan;
        camera.focalLength = 15.0f;

        Stream resLoader;

        SwapChainT swapChain(3);
        std::queue<RenderingTask> tasks;
        {
            Stream copyStream;
            window.bindBackBuffer(copyStream.get());
            auto lum = MemorySpan<float>();
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

