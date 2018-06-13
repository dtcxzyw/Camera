#include "kernel.hpp"
#include <IO/Image.hpp>
#include <Core/Environment.hpp>
#include <Interaction/SwapChain.hpp>
#include <Core/CompileBegin.hpp>
#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <Core/CompileEnd.hpp>
#include <Interaction/SoftwareRenderer.hpp>
#include <Interaction/D3D11.hpp>

using namespace std::chrono_literals;

class App final : Uncopyable {
private:
    float mLight = 65.0f, mR = 20.0f;
    std::unique_ptr<StaticMeshData> mBox, mModel;
    MemorySpan<SphereDesc> mSpheres;
    std::unique_ptr<RenderingContext> mMc;
    std::unique_ptr<RenderingContext> mSc;
    std::shared_ptr<BuiltinCubeMap<RGBA>> mEnvMap;
    std::shared_ptr<BuiltinSampler<RGBA>> mEnvMapSampler;
    DisneyBRDFArg mArg;
    PinholeCamera mCamera;
    Uniform mOld;
    RGB mColor;

    static void setStyle() {
        ImGui::StyleColorsDark();
        auto& style = ImGui::GetStyle();
        style.Alpha = 0.8f;
        style.ChildBorderSize = 0.0f;
        style.ChildRounding = 0.0f;
        style.FrameBorderSize = 0.0f;
        style.FrameRounding = 0.0f;
        style.GrabRounding = 0.0f;
        style.PopupBorderSize = 0.0f;
        style.PopupRounding = 0.0f;
        style.ScrollbarRounding = 0.0f;
        style.WindowRounding = 0.0f;
        style.WindowBorderSize = 0.0f;
        style.AntiAliasedFill = true;
        style.AntiAliasedLines = true;
    }

    void renderGUI(D3D11Window& window) {
        window.newFrame();
        ImGui::Begin("Debug");
        ImGui::SetWindowPos({0, 0});
        ImGui::SetWindowSize({500, 580});
        ImGui::SetWindowFontScale(1.5f);
        ImGui::Text("vertices: %d, triangles: %d\n", static_cast<int>(mModel->vert.size()),
            static_cast<int>(mModel->index.size()));
        ImGui::Text("triNum: %u\n", *mMc->triContext.triNum);
        ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
        auto&& monitor = DeviceMonitor::get();
        ImGui::Text("Memory %.1f%", monitor.getMemoryFreeSize() * 100.0f / monitor.getMemoryTotalSize());
        ImGui::Text("focal length %.1f ", mCamera.toFocalLength());
        ImGui::SliderFloat("fov", &mCamera.fov, 1.0f, 120.0f, "%.1f");
        ImGui::SliderFloat("light", &mLight, 0.0f, 100.0f);
        ImGui::SliderFloat("lightRadius", &mR, 0.0f, 40.0f);
        mColor = clamp(mColor, RGB(0.01f), RGB(0.999f));
        ImGui::ColorEdit3("baseColor", &mColor[0], ImGuiColorEditFlags_Float);

        #define ARG(name)\
 mArg.##name=clamp(mArg.##name,0.01f,0.999f);\
 ImGui::SliderFloat(#name, &mArg.##name, 0.01f, 0.999f);
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
        ImGui::Render();
    }

    Uniform getUniform(float, const vec2 mul) const {
        static Vector cp = {10.0f, 0.0f, 0.0f}, mid = {-100000.0f, 0.0f, 0.0f};
        const auto V = glm::lookAt(cp, mid, {0.0f, 1.0f, 0.0f});
        auto M = glm::scale(glm::mat4{}, Vector(5.0f));
        M = glm::rotate(M, half_pi<float>(), {0.0f, 1.0f, 0.0f});
        constexpr auto step = 50.0f;
        const auto off = ImGui::GetIO().DeltaTime * step;
        if (ImGui::IsKeyDown('W'))cp.x -= off;
        if (ImGui::IsKeyDown('S'))cp.x += off;
        if (ImGui::IsKeyDown('A'))cp.z += off;
        if (ImGui::IsKeyDown('D'))cp.z -= off;
        Uniform u;
        u.mul = mul;
        u.cameraTransform = Transform(V);
        u.skyTransform = Transform(mat3(V));
        u.invCameraTransform = inverse(u.cameraTransform);
        u.modelTransform = Transform(M);
        u.arg = mArg;
        u.arg.baseColor = Spectrum(mColor);
        u.cp = Point(cp);
        u.light = {u.cp + Vector{0.0f, 4.0f, 0.0f}, Spectrum{mLight}};
        u.sampler = mEnvMapSampler->toRef();
        return u;
    }

    using SharedFrame = std::shared_ptr<FrameBuffer>;

    struct RenderingTask {
        Future future;
        SharedFrame frame;

        RenderingTask(const Future& fut, const SharedFrame& fbo)
            : future(fut), frame(fbo) {}
    };

    static float getTime() {
        const double t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return static_cast<float>(t * 1e-9);
    }

    static constexpr auto enableTriCache = true;

    auto addTask(SharedFrame frame, const uvec2 size, float* lum) {
        static auto last = getTime();
        const auto now = getTime();
        const auto converter = mCamera.toRasterPos(size, 1.0f, 200.0f);
        auto buffer = std::make_unique<CommandBuffer>();
        if (frame->size != size) {
            mMc->triContext.reset(mModel->index.size(), 65536U, false, enableTriCache);
            mSc->triContext.reset(mBox->index.size(), 65536U, false, enableTriCache);
        }
        frame->resize(size);
        {
            auto uniform = getUniform(now - last, converter.mul);
            if (uniform.cp != mOld.cp)mMc->vertCounter.count();
            if (uniform.mul != mOld.mul) {
                mMc->vertCounter.count();
                mSc->vertCounter.count();
            }
            mOld = uniform;
            const auto uni = buffer->allocConstant<Uniform>();
            buffer->memcpy(uni, [uniform](auto call) {
                call(&uniform);
            });
            kernel(*mModel, *mMc, *mBox, *mSc, mSpheres, uni, *frame, lum, converter, *buffer);
        }
        last = now;
        renderGUI(D3D11Window::get());
        SoftwareRenderer::get().render(*buffer, *frame->postRT);
        return RenderingTask{Environment::get().submit(std::move(buffer)), frame};
    }

    void uploadSpheres() {
        const SphereDesc sphere[] = {{{0.0f, 3.0f, 10.0f}, 5.0f}, {{0.0f, 0.0f, 13.0f}, 3.0f}};
        mSpheres = MemorySpan<SphereDesc>(std::size(sphere));
        checkError(cudaMemcpy(mSpheres.begin(), sphere, sizeof(sphere), cudaMemcpyHostToDevice));
    }

public:
    void run() {
        auto&& window = D3D11Window::get();
        window.show(true);
        setStyle();
        ImGui::GetIO().WantCaptureKeyboard = true;

        auto&& env = Environment::get();
        env.init(AppType::Online, GraphicsInteroperability::D3D11);

        {
            Stream resLoader;

            SoftwareRenderer::get().init(resLoader);

            uploadSpheres();
            //StaticMesh model("Res/mitsuba/mitsuba-sphere.obj");
            StaticMesh model("Res/dragon.obj");
            mModel = std::make_unique<StaticMeshData>(model, resLoader);
            mMc = std::make_unique<RenderingContext>();
            mMc->triContext.reset(mModel->index.size(), 65536U, false, enableTriCache);

            StaticMesh box("Res/cube.obj");
            mBox = std::make_unique<StaticMeshData>(box, resLoader);
            mSc = std::make_unique<RenderingContext>();
            mSc->triContext.reset(mBox->index.size(), 65536U, false, enableTriCache);

            mEnvMap = loadCubeMap([](const size_t id) {
                const char* table[] = {"right", "left", "top", "bottom", "back", "front"};
                return std::string("Res/skybox/") + table[id] + ".jpg";
            }, resLoader);
            mEnvMapSampler = std::make_shared<BuiltinSampler<RGBA>>(mEnvMap->get());
        }

        mColor = RGB{220.0f, 223.0f, 227.0f} / 255.0f;

        std::queue<RenderingTask> tasks;

        {
            Stream copyStream;
            window.bindBackBuffer(copyStream.get());
            auto lum = MemorySpan<float>(1);

            constexpr auto queueSize = 3;

            {
                const auto size = window.size();
                for (auto i = 0; i < queueSize; ++i) {
                    const auto tb = Clock::now();
                    tasks.push(addTask(std::make_shared<FrameBuffer>(), size, lum.begin()));
                    const auto te = Clock::now();
                    const auto delta = (te - tb).count() * 1e-6f;
                    printf("build time:%.3f ms\n", delta);
                }
            }

            while (window.update()) {
                const auto size = window.size();
                if (size.x == 0 || size.y == 0) {
                    Environment::get().yield();
                    continue;
                }

                tasks.front().future.sync();
                auto frame = std::move(tasks.front().frame);
                tasks.pop();

                if (frame->size == size) {
                    window.present(frame->postRT->get());
                    window.swapBuffers();
                }

                const auto tb = Clock::now();
                tasks.push(addTask(std::move(frame), size, lum.begin()));
                const auto te = Clock::now();
                const auto delta = (te - tb).count() * 1e-6f;
                printf("build time:%.3f ms\n", delta);
            }
            window.unbindBackBuffer();
        }

        env.uninit();
        SoftwareRenderer::get().uninit();
    }
};

int main() {
    App app;
    app.run();
    return 0;
}
