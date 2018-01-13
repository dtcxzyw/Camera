#include "kernel.hpp"
#include <cstdio>
#include <IO/Image.hpp>
#include <thread>
using namespace std::chrono_literals;

auto light=5.0f,r=20.0f;
StaticMesh model;
std::shared_ptr<BuiltinCubeMap<RGBA>> envMap;
std::shared_ptr<BuiltinSampler<RGBA>> envMapSampler;
DisneyBRDFArg arg;
Camera camera;

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
    ImGui::Text("FOV %.1f ",degrees(camera.toFOV()));
    ImGui::SliderFloat("focal length",&camera.focalLength,1.0f,500.0f,"%.1f");
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
    Arg(sheen);
    Arg(sheenTint);
    Arg(clearcoat);
    Arg(clearcoatGloss);
#undef Arg
    ImGui::End();
    window.renderGUI();
}

Uniform getUniform(float w,float h,float delta) {
    static vec3 cp = { 3.0f,0.0f,0.0f }, lp = { 10.0f,4.0f,0.0f }, mid = { -100000.0f,0.0f,0.0f };
    auto V = lookAt(cp,mid, { 0.0f,1.0f,0.0f });
    glm::mat4 M = scale(mat4{}, vec3(0.01f));
    M = rotate(M, 0.5f, { 0.0f,1.0f,0.0f });
    constexpr auto step = 10.0f;
    auto off = ImGui::GetIO().DeltaTime * step;
    if (ImGui::IsKeyPressed(GLFW_KEY_W))cp.x -= off;
    if (ImGui::IsKeyPressed(GLFW_KEY_S))cp.x += off;
    if (ImGui::IsKeyPressed(GLFW_KEY_A))cp.z -= off;
    if (ImGui::IsKeyPressed(GLFW_KEY_D))cp.z += off;
    Uniform u;
    u.M = M;
    u.V = V;
    u.invM = mat3(transpose(inverse(M)));
    u.lc = vec3(light);
    u.arg = arg;
    u.cp = cp;
    u.lp = lp;
    u.r = r;
    u.sampler = envMapSampler->toSampler();
    return u;
}

using SwapChain_t = SwapChain<FrameBufferCPU>;

auto addTask(DispatchSystem& system,SwapChain_t::SharedFrame frame,uvec2 size,float* lum) {
    static float last = glfwGetTime();
    float now = glfwGetTime();
    auto uniform = getUniform(size.x,size.y,now-last);
    last = now;
    auto buffer=std::make_unique<CommandBuffer>();
    frame->resize(size.x,size.y);
    auto uni = buffer->allocConstant<Uniform>();
    buffer->memcpy(uni, [uniform](auto call) {call(&uniform); });
    kernel(model.mVert,model.mIndex,uni,*frame,lum,camera.toRasterPos(size),*buffer);
    return std::make_pair(system.submit(std::move(buffer)),frame);
}

int main() {
    getEnvironment().init();
    try {
        camera.near = 1.0f;
        camera.far = 5.0f;
        camera.filmAperture = { 0.980f,0.735f };
        camera.mode = Camera::FitResolutionGate::Overscan;
        camera.focalLength = 15.0f;
        Stream resLoader;
        model.load("Res/cube.obj");
        envMap = loadCubeMap([](size_t id) {
            const char* table[] = {"right","left","top","bottom","back","front"};
            return std::string("Res/skybox/")+table[id]+".jpg";
        }, resLoader);
        envMapSampler = std::make_shared<BuiltinSampler<RGBA>>(envMap->get());
        arg.baseColor = vec3{ 244,206,120 }/255.0f;
        //arg.edgeTint = vec3{ 254,249,205 } / 255.0f;
        IMGUIWindow window;
        setUIStyle();
        ImGui::GetIO().WantCaptureKeyboard = true;

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
                    system.update(1ms);
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
        
        envMapSampler.reset();
        envMap.reset();
    }
    catch (const std::exception& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}

