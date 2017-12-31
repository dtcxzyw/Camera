#include "kernel.hpp"
#include <cstdio>
#include <PBR/PhotorealisticRendering.hpp>
#include <thread>
using namespace std::chrono_literals;

auto f = 50.0f, light=5.0f,r=20.0f;
StaticMesh model;
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
    ImGui::SetWindowSize({ 500,530 });
    ImGui::SetWindowFontScale(1.5f);
    ImGui::Text("vertices %d, triangles: %d\n", static_cast<int>(model.mVert.size()),
        static_cast<int>(model.mIndex.size()));
    ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
    ImGui::SliderFloat("focal length",&f,15.0f,500.0f,"%.1f");
    ImGui::SliderFloat("light", &light, 0.0f, 10.0f);
    ImGui::SliderFloat("lightRadius", &r, 0.0f, 20.0f);

#define Color(name)\
arg.##name=clamp(arg.##name,vec3(0.01f),vec3(0.999f));\
ImGui::ColorEdit3(#name,&arg.##name[0],ImGuiColorEditFlags_Float);\

    Color(baseColor);
    //Color(edgeTint);
#undef Color

#define Arg(name)\
 arg.##name=clamp(arg.##name,0.01f,0.999f);\
 ImGui::SliderFloat(#name, &arg.##name, 0.01f, 0.999f);\

    Arg(metallic);
    Arg(subsurface);
    Arg(specular);
    Arg(roughness);
    Arg(specularTint);
    Arg(anisotropic);
    //Arg(anisotropy);
    Arg(sheen);
    Arg(sheenTint);
    Arg(clearcoat);
    Arg(clearcoatGloss);
    //Arg(cavity);
    //Arg(smoothness);
    //Arg(reflectance);
#undef Arg
    ImGui::End();
    window.renderGUI();
}

Uniform getUniform(float w,float h,float delta) {
    vec3 cp = { 10.0f,0.0f,0.0f }, lp = { 10.0f,4.0f,0.0f };
    auto V = lookAt(cp, { 0.0f,0.0f,0.0f }, { 0.0f,1.0f,0.0f });
    static glm::mat4 M = scale(glm::mat4{}, vec3(1.0f));
    auto fov = toFOV(36.0f*24.0f, f);
    glm::mat4 P = perspectiveFov(fov, w, h, 1.0f, 20.0f);
    M = rotate(M, delta*0.2f, { 0.0f,1.0f,0.0f });
    Uniform u;
    u.VP = P * V;
    u.M = M;
    u.invM = mat3(transpose(inverse(M)));
    u.lc = vec3(light);
    u.arg = arg;
    u.cp = cp;
    u.lp = lp;
    u.r = r;
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
        model.load("Res/mitsuba/mitsuba-sphere.obj");
        arg.baseColor = vec3{ 244,206,120 }/255.0f;
        //arg.edgeTint = vec3{ 254,249,205 } / 255.0f;
        IMGUIWindow window;
        setUIStyle();
        SwapChain_t swapChain(5);
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

