#include "kernel.hpp"
#include <cstdio>
#include <IO/Image.hpp>
#include <thread>
using namespace std::chrono_literals;

auto light=65.0f,r=20.0f;
StaticMesh box,model;
TriangleRenderingHistory mh;
TriangleRenderingHistory sh;
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
    ImGui::SetWindowSize({ 500,550 });
    ImGui::SetWindowFontScale(1.5f);
    ImGui::Text("vertices: %d, triangles: %d\n", static_cast<int>(model.vert.size()),
        static_cast<int>(model.index.size()));
    ImGui::Text("triNum: %d\n", static_cast<int>(mh.triNum));
    ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
    ImGui::Text("FOV %.1f ",degrees(camera.toFov()));
    ImGui::SliderFloat("focal length",&camera.focalLength,1.0f,500.0f,"%.1f");
    ImGui::SliderFloat("light", &light, 0.0f, 100.0f);
    ImGui::SliderFloat("lightRadius", &r, 0.0f, 20.0f);

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

Uniform getUniform(float, const vec2 mul) {
    static vec3 cp = { 10.0f,0.0f,0.0f }, lp = { 10.0f,4.0f,0.0f }, mid = { -100000.0f,0.0f,0.0f };
    const auto V = lookAt(cp,mid, { 0.0f,1.0f,0.0f });
    auto M= scale(mat4{}, vec3(5.0f));
    M = rotate(M, half_pi<float>(), { 0.0f,1.0f,0.0f });
    constexpr auto step = 50.0f;
    const auto off = ImGui::GetIO().DeltaTime * step;
    if (ImGui::IsKeyPressed(GLFW_KEY_W))cp.x -= off;
    if (ImGui::IsKeyPressed(GLFW_KEY_S))cp.x += off;
    if (ImGui::IsKeyPressed(GLFW_KEY_A))cp.z -= off;
    if (ImGui::IsKeyPressed(GLFW_KEY_D))cp.z += off;
    Uniform u;
    u.mul = mul;
    u.Msky = {};
    u.M = M;
    u.V = V;
    u.invM = mat3(transpose(inverse(u.M)));
    u.lc = vec3(light);
    u.arg = arg;
    u.cp = cp;
    u.lp = lp;
    u.r = r;
    u.sampler = envMapSampler->toSampler();
    return u;
}

using SwapChainT = SwapChain<FrameBufferCPU>;
struct RenderingTask {
    Future future;
    SwapChainT::SharedFrame frame;
    RC8::Block block;
    RenderingTask(const Future& fut, const SwapChainT::SharedFrame& fbo, const RC8::Block blockInfo)
    :future(fut), frame(fbo),block(blockInfo){}
};

auto addTask(DispatchSystem& system,SwapChainT::SharedFrame frame,uvec2 size,
    float* lum,RC8& cache) {
    static float last = glfwGetTime();
    const float now = glfwGetTime();
    const auto converter = camera.toRasterPos(size);
    auto uniform = getUniform(now-last,converter.mul);
    last = now;
    auto buffer=std::make_unique<CommandBuffer>();
    if (frame->size != size) {
        mh.reset(model.index.size(),cache.blockSize()*3,true);
        cache.reset();
        sh.reset(box.index.size());
    }
    frame->resize(size);
    auto uni = buffer->allocConstant<Uniform>();
    auto block = cache.pop(*buffer);
    uniform.cache = block.toBlock();
    buffer->memcpy(uni, [uniform](auto call) {call(&uniform); });
    kernel(model,mh,box,sh,uni,*frame,lum,converter,*buffer);
    return RenderingTask{ system.submit(std::move(buffer)),frame,block};
}

int main() {
    getEnvironment().init();
    try {
        camera.near = 1.0f;
        camera.far = 200.0f;
        camera.filmAperture = { 0.980f,0.735f };
        camera.mode = Camera::FitResolutionGate::Overscan;
        camera.focalLength = 15.0f;

        Stream resLoader;
        //model.load("Res/mitsuba/mitsuba-sphere.obj",resLoader);
        model.load("Res/dragon.obj",resLoader);
        RC8 cache(model.index.size(),30);
        mh.reset(model.index.size(),cache.blockSize()*3,true);

        box.load("Res/cube.obj",resLoader);
        sh.reset(box.index.size());
        
        envMap = loadCubeMap([](size_t id) {
            const char* table[] = {"right","left","top","bottom","back","front"};
            return std::string("Res/skybox/")+table[id]+".jpg";
        }, resLoader);
        //envMap = loadRGBA("Res/Helipad_Afternoon/LA_Downtown_Afternoon_Fishing_B_8k.jpg",resLoader);
        envMapSampler = std::make_shared<BuiltinSampler<RGBA>>(envMap->get());
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
                        tasks.push(addTask(system, swapChain.pop(), size, lum.begin(),cache));
                    if (!tasks.empty() && tasks.front().future.finished()) {
                        frame = tasks.front().frame;
                        cache.push(tasks.front().block);
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
        
        envMapSampler.reset();
        envMap.reset();
    }
    catch (const std::runtime_error& e) {
        puts("Catched an error:");
        puts(e.what());
        system("pause");
    }
    return 0;
}

