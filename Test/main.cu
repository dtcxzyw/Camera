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

auto light=65.0f,r=20.0f;
StaticMesh box,model;
std::unique_ptr<RC8> cache;
DataViewer<vec4> spheres;
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

void renderGUI(D3D11Window& window) {
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
    ImGui::SliderFloat("lightRadius", &r, 0.0f, 40.0f);

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
    static vec3 cp = { 10.0f,0.0f,0.0f }, mid = { -100000.0f,0.0f,0.0f };
    const auto V = lookAt(cp,mid, { 0.0f,1.0f,0.0f });
    auto M= scale(mat4{}, vec3(5.0f));
    M = rotate(M, half_pi<float>(), { 0.0f,1.0f,0.0f });
    constexpr auto step = 50.0f;
    const auto off = ImGui::GetIO().DeltaTime * step;
    if (ImGui::IsKeyPressed('W'))cp.x -= off;
    if (ImGui::IsKeyPressed('S'))cp.x += off;
    if (ImGui::IsKeyPressed('A'))cp.z += off;
    if (ImGui::IsKeyPressed('D'))cp.z -= off;
    Uniform u;
    u.mul = mul;
    u.Msky = {};
    u.M = M;
    u.V = V;
    u.invV = inverse(u.V);
    u.normalInvV = mat3(transpose(u.V));
    u.normalMat = mat3(transpose(inverse(u.M)));
    u.lc = vec3(light);
    u.arg = arg;
    u.cp = cp;
    u.lp = cp+vec3{0.0f,4.0f,0.0f};
    u.r2 = r*r;
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

constexpr auto enableSAA = true;

float getTime() {
    const double t=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return static_cast<float>(t * 1e-9);
}

auto addTask(SwapChainT::SharedFrame frame, const uvec2 size,float* lum) {
    static auto last = getTime();
    const auto now = getTime();
    const auto converter = camera.toRasterPos(size);
    last = now;
    auto buffer=std::make_unique<CommandBuffer>();
    if (frame->size != size) {
        mh.reset(model.index.size(), cache->blockSize() * 3, enableSAA);
        cache->reset();
        sh.reset(box.index.size());
    }
    frame->resize(size);
    auto block = cache->pop(*buffer);
    {
        auto uniform = getUniform(now - last, converter.mul);
        auto uni = buffer->allocConstant<Uniform>();
        uniform.cache = block.toBlock();
        buffer->memcpy(uni, [uniform](auto call) {call(&uniform); });
        kernel(model, mh, box, sh, spheres, uni, *frame, lum, converter, *buffer);
    }
    return RenderingTask{ getEnvironment().submit(std::move(buffer)),frame,block};
}

void uploadSpheres() {
    vec4 sphere[] = {{0.0f,3.0f,10.0f,5.0f},{0.0f,0.0f,13.0f,3.0f}};
    spheres = DataViewer<vec4>(std::size(sphere));
    checkError(cudaMemcpy(spheres.begin(),sphere,sizeof(sphere),cudaMemcpyHostToDevice));
}

int main() {
    auto&& window=getD3D11Window();
    window.show(true);
    setUIStyle();
    ImGui::GetIO().WantCaptureKeyboard = true;

    auto&& env = getEnvironment();
    env.init(GraphicsInteroperability::D3D11);

    try {
        camera.near = 1.0f;
        camera.far = 200.0f;
        camera.filmAperture = { 0.980f,0.735f };
        camera.mode = Camera::FitResolutionGate::Overscan;
        camera.focalLength = 15.0f;

        {
            Stream resLoader;
            uploadSpheres();
            //model.load("Res/mitsuba/mitsuba-sphere.obj",resLoader);
            model.load("Res/dragon.obj", resLoader);
            cache = std::make_unique<RC8>(model.index.size());
            mh.reset(model.index.size(), cache->blockSize() * 3, enableSAA);

            box.load("Res/cube.obj", resLoader);
            sh.reset(box.index.size());

            envMap = loadCubeMap([](size_t id) {
                const char* table[] = { "right","left","top","bottom","back","front" };
                return std::string("Res/skybox/") + table[id] + ".jpg";
            }, resLoader);
            //envMap = loadRGBA("Res/Helipad_Afternoon/LA_Downtown_Afternoon_Fishing_B_8k.jpg",resLoader);
            envMapSampler = std::make_shared<BuiltinSampler<RGBA>>(envMap->get());
        }

        arg.baseColor = vec3{220,223,227}/255.0f;

        SwapChainT swapChain(3);
        std::queue<RenderingTask> tasks;
        {
            auto lum = DataViewer<float>(1);
            while (window.update()) {
                const auto size = window.size();
                if (size.x == 0 || size.y == 0) {
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                SwapChainT::SharedFrame frame;
                while (true) {
                    if (!swapChain.empty())
                        tasks.push(addTask(swapChain.pop(), size, lum.begin()));
                    if (!tasks.empty() && tasks.front().future.finished()) {
                        frame = tasks.front().frame;
                        cache->push(tasks.front().block);
                        tasks.pop();
                        break;
                    }
                }
                window.present(frame->image);
                renderGUI(window);
                window.swapBuffers();
                swapChain.push(std::move(frame));
            }
        }
        
        env.uninit();
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

