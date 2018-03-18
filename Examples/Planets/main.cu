#include "kernel.hpp"
#include <thread>
#include <Base/Environment.hpp>
#include <Interaction/SwapChain.hpp>
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <Base/CompileEnd.hpp>
#include <Interaction/SoftwareRenderer.hpp>
#include <random>

using namespace std::chrono_literals;

class System final:Uncopyable {
private:
    struct Planet final {
        dvec3 pos,vel;
        double r;
    };
    std::vector<Planet> mPlanets;
    double mEdge;

    static float mass(const double r) {
        const auto v = 0.75*pi<double>()*r*r*r;
        return v*1e11;
    }

    void updateStep(const double delta) {
        constexpr auto g= 6.67259e-11;
        std::vector<double> pmass(mPlanets.size());
        for (size_t i = 0; i < mPlanets.size(); ++i)
            pmass[i] = mass(mPlanets[i].r);
        for (size_t i = 0; i < mPlanets.size(); ++i){
            dvec3 acc{};
            for (size_t j = 0; j < mPlanets.size(); ++j)
                if (i != j){
                    const auto deltaPos = dvec3(mPlanets[j].pos) - dvec3(mPlanets[i].pos);
                    acc += normalize(deltaPos)*(pmass[j] / length2(deltaPos));
                }
            acc *= g;
            mPlanets[i].vel += acc * delta;
            mPlanets[i].pos += mPlanets[i].vel*delta;
        }
        for (size_t i = 0; i < mPlanets.size(); ++i) {
            for (auto j = i + 1; j < mPlanets.size(); ++j) {
                const auto deltaPos = dvec3(mPlanets[j].pos) - dvec3(mPlanets[i].pos);
                const auto dis = length(deltaPos);
                const auto sr = mPlanets[i].r + mPlanets[j].r;
                if(dis<sr) {
                    const auto deltaDis=sr-dis;
                    const auto m1 = mass(mPlanets[i].r), m2 = mass(mPlanets[j].r);
                    const auto dir = normalize(deltaPos);
                    mPlanets[i].pos -= dir * (deltaDis*m2 / (m1 + m2));
                    mPlanets[j].pos += dir * (deltaDis*m1 / (m1 + m2));
                }
            }
            mPlanets[i].pos = clamp(mPlanets[i].pos, -mEdge, mEdge);
        }
    }
public:
    System(const double edge, const size_t num) :mPlanets(num), mEdge(edge) {
        std::mt19937_64 gen(Clock::now().time_since_epoch().count());
        const std::uniform_real_distribution<double> dim(-edge,edge);
        const std::uniform_real_distribution<double> vel(-1.0, 1.0);
        const std::uniform_real_distribution<double> radius(0.2, 1.0);
        for(auto&& p:mPlanets) {
            p.pos = { dim(gen),dim(gen),dim(gen) };
            p.vel = { vel(gen),vel(gen),vel(gen) };
            p.r = radius(gen);
        }
    }
    void update(double delta) {
        constexpr auto step = 1e-4;
        while(delta>step) {
            updateStep(step);
            delta -= step;
        }
        updateStep(delta);
    }
    MemoryRef<vec4> getDrawData(CommandBuffer& buffer) const {
        std::vector<vec4> data(mPlanets.size());
        for (size_t i = 0; i < mPlanets.size(); ++i)
            data[i] = { mPlanets[i].pos,mPlanets[i].r };
        auto buf = buffer.allocBuffer<vec4>(mPlanets.size());
        buffer.memcpy(buf, [holder=std::move(data)](auto&& call){
            call(holder.data());
        });
        return buf;
    }
};

class App final : Uncopyable {
private:
    Camera mCamera;

    void setUIStyle() {
        ImGui::StyleColorsDark();
        auto& style = ImGui::GetStyle();
        style.Alpha = 0.8f;
        style.AntiAliasedFill = true;
        style.AntiAliasedLines = true;
    }

    void renderGUI(D3D11Window& window) {
        window.newFrame();
        ImGui::Begin("Debug");
        ImGui::SetWindowPos({0, 0});
        ImGui::SetWindowSize({500, 150});
        ImGui::SetWindowFontScale(1.5f);
        ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
        ImGui::Text("FOV %.1f ", degrees(mCamera.toFov()));
        ImGui::SliderFloat("focal length", &mCamera.focalLength, 1.0f, 500.0f, "%.1f");
        ImGui::End();
        ImGui::Render();
    }

    Uniform getUniform(const float delta) {
        static vec3 cp = {80.0f, 0.0f, 0.0f}, mid = {-100000.0f, 0.0f, 0.0f};
        const auto V = lookAt(cp, mid, {0.0f, 1.0f, 0.0f});
        constexpr auto step = 50.0f;
        const auto off = delta * step;
        if (ImGui::IsKeyPressed('W'))cp.x -= off;
        if (ImGui::IsKeyPressed('S'))cp.x += off;
        if (ImGui::IsKeyPressed('A'))cp.z += off;
        if (ImGui::IsKeyPressed('D'))cp.z -= off;
        Uniform u;
        u.V = V;
        return u;
    }

    using SharedFrame= std::shared_ptr<FrameBufferCPU>;

    struct RenderingTask {
        Future future;
        SharedFrame frame;

        RenderingTask(const Future& fut, const SharedFrame& fbo)
            : future(fut), frame(fbo) {}
    };

    double getTime() {
        const double t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return t * 1e-9;
    }

    auto addTask(System& system,SharedFrame frame, const uvec2 size) {
        static auto last = getTime();
        const auto now = getTime();
        const auto converter = mCamera.toRasterPos(size);
        auto buffer = std::make_unique<CommandBuffer>();
        frame->resize(size);
        {
            auto uniform = getUniform(static_cast<float>(now - last));
            auto uni = buffer->allocConstant<Uniform>();
            buffer->memcpy(uni, [uniform](auto call) {
                call(&uniform);
            });
            system.update(now - last);
            kernel(system.getDrawData(*buffer), uni, *frame, converter, *buffer);
        }
        last = now;
        renderGUI(D3D11Window::get());
        SoftwareRenderer::get().render(*buffer, *frame->postRT);
        return RenderingTask{Environment::get().submit(std::move(buffer)), frame};
    }

public:
    void run() {
        auto&& window = D3D11Window::get();
        window.show(true);
        setUIStyle();
        ImGui::GetIO().WantCaptureKeyboard = true;

        auto&& env = Environment::get();
        env.init(AppType::Online,GraphicsInteroperability::D3D11);

        mCamera.near = 1.0f;
        mCamera.far = 200.0f;
        mCamera.filmAperture = {0.980f, 0.735f};
        mCamera.mode = Camera::FitResolutionGate::Overscan;
        mCamera.focalLength = 15.0f;

        {
            Stream resLoader;

            SoftwareRenderer::get().init(resLoader);
        }

        std::queue<RenderingTask> tasks;

        System system(40.0, 40);

        {
            Stream copyStream;
            window.bindBackBuffer(copyStream.get());

            constexpr auto queueSize = 3;

            {
                const auto size = window.size();
                for (auto i = 0; i < queueSize; ++i) {
                    tasks.push(addTask(system,std::make_shared<FrameBufferCPU>(), size));
                }
            }

            while (window.update()) {
                const auto size = window.size();
                if (size.x == 0 || size.y == 0) {
                    std::this_thread::sleep_for(1ms);
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
                tasks.push(addTask(system,std::move(frame), size));
                const auto te = Clock::now();
                const auto delta = (te - tb).count() * 1e-6f;
                printf("build time:%.3f ms\n",delta);
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
