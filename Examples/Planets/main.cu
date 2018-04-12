#include "kernel.hpp"
#include <thread>
#include <Core/Environment.hpp>
#include <Interaction/SwapChain.hpp>
#include <Core/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <Core/CompileEnd.hpp>
#include <Interaction/SoftwareRenderer.hpp>

using namespace std::chrono_literals;

constexpr auto au = 149597870.7;
float scalePow =9.35f; //10.8f;
float timeFac = 1.0f;
int timePow = 0;

class System final:Uncopyable {
private:
    struct Planet final {
        dvec3 pos,vel;
        double r,mass;
    };
    std::vector<Planet> mPlanets;

    void updateStep(double delta) {
        delta *= timeFac;
        delta *= pow(10.0, timePow);
        constexpr auto g= 6.67259e-11;
        for (size_t i = 0; i < mPlanets.size(); ++i){
            dvec3 acc{};
            for (size_t j = 0; j < mPlanets.size(); ++j)
                if (i != j){
                    const auto deltaPos = dvec3(mPlanets[j].pos) - dvec3(mPlanets[i].pos);
                    acc += normalize(deltaPos)*(mPlanets[j].mass / length2(deltaPos));
                }
            mPlanets[i].vel += acc * (g*delta);
            mPlanets[i].pos += mPlanets[i].vel*delta;
        }
    }
    void check() {
        for (size_t i = 0; i < mPlanets.size(); ++i) {
            for (auto j = i + 1; j < mPlanets.size(); ++j) {
                const auto deltaPos = dvec3(mPlanets[j].pos) - dvec3(mPlanets[i].pos);
                const auto dis = length(deltaPos);
                const auto sr = mPlanets[i].r + mPlanets[j].r;
                if (dis<sr) {
                    const auto deltaDis = sr - dis;
                    const auto m1 = mPlanets[i].mass, m2 = mPlanets[j].mass;
                    const auto dir = normalize(deltaPos);
                    mPlanets[i].pos -= dir * (deltaDis*m2 / (m1 + m2));
                    mPlanets[j].pos += dir * (deltaDis*m1 / (m1 + m2));
                }
            }
        }
    }
public:
    System() {
        const auto insert = [this](double radius,double mass,double pos,double vel) {
            Planet p;
            p.pos = {0.0, pos * au * 1e3, 0.0};
            p.vel = {0.0, 0.0, vel * 1e3};
            p.mass = mass;
            p.r = radius * 1e3;
            mPlanets.emplace_back(p);
        };
        //see https://en.wikipedia.org/wiki/List_of_gravitationally_rounded_objects_of_the_Solar_System
        insert(696342, 1.9855e30, 0.0, 0.0);

        insert(2439.64, 3.302e23, 0.38709893, 47.8725);
        insert(6051.59, 4.8690e24, 0.72333199, 35.0214);
        insert(6378.1, 5.972e24, 1.00000011, 29.7859);
        insert(3397.00, 6.4191e23, 1.52366231, 24.1309);
        insert(71492.68, 1.8987e27, 5.20336301, 13.0697);
        insert(60267.14, 5.6851e26, 9.53707032, 9.6724);
        insert(25557.25, 8.6849e25, 19.19126393, 6.8352);
        insert(24766.36, 1.0244e26, 30.06896348, 5.4778);

        constexpr auto mdis = 384399.0 / au + 1.00000011;
        insert(1737.1, 7.3477e22, mdis, 1.022 + 29.7859);
    }

    void update(double delta) {
        constexpr auto step = 2e-6;
        while(delta>step) {
            updateStep(step);
            delta -= step;
        }
        updateStep(delta);
        check();
    }
    Span<vec4> getDrawData(CommandBuffer& buffer) {
        const auto fac = pow(0.1, scalePow);
        std::vector<vec4> data(mPlanets.size());
        for (size_t i = 0; i < mPlanets.size(); ++i){
            data[i] = { mPlanets[i].pos*fac,fmax(0.1f,mPlanets[i].r*fac) };
        }
        auto buf = buffer.allocBuffer<vec4>(data.size());
        buffer.memcpy(buf, [holder = std::move(data)](auto&& call){
            call(holder.data());
        });
        return buf;
    }
    vec3 getPlanetPos(const size_t id) const {
        return mPlanets[id].pos*pow(0.1, scalePow);
    }
};

class App final : Uncopyable {
private:
    Camera mCamera;

    void setStyle() {
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
        ImGui::SetWindowSize({500, 200});
        ImGui::SetWindowFontScale(1.5f);
        ImGui::Text("FPS %.1f ", ImGui::GetIO().Framerate);
        ImGui::Text("FOV %.1f ", degrees(mCamera.toFov()));
        ImGui::SliderFloat("focal length", &mCamera.focalLength, 1.0f, 500.0f, "%.1f");
        ImGui::SliderFloat("scale",&scalePow,0.0f,11.0f);
        ImGui::SliderFloat("time fac", &timeFac, 1.0f, 9.9f);
        ImGui::SliderInt("time pow", &timePow, 0, 10);
        ImGui::End();
        ImGui::Render();
    }

    static Uniform getUniform(const vec3 earth) {
        static vec3 cp = {120.0f, 0.0f, 0.0f};
        const auto mid = ImGui::IsKeyDown('F') ? earth : vec3{ -100000.0f, 0.0f, 0.0f };
        const auto V = lookAt(cp, mid, {0.0f, 1.0f, 0.0f});
        Uniform u;
        u.V = V;
        return u;
    }

    using SharedFrame= std::shared_ptr<FrameBuffer>;

    struct RenderingTask {
        Future future;
        SharedFrame frame;

        RenderingTask(const Future& fut, const SharedFrame& fbo)
            : future(fut), frame(fbo) {}
    };

    static double getTime() {
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
            auto uniform = getUniform(system.getPlanetPos(3));
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
        setStyle();
        ImGui::GetIO().WantCaptureKeyboard = true;

        auto&& env = Environment::get();
        env.init(AppType::Online,GraphicsInteroperability::D3D11);

        mCamera.near = 1.0f;
        mCamera.far = 250.0f;
        mCamera.filmAperture = {0.980f, 0.735f};
        mCamera.mode = Camera::FitResolutionGate::Overscan;
        mCamera.focalLength = 15.0f;

        {
            Stream resLoader;

            SoftwareRenderer::get().init(resLoader);
        }

        std::queue<RenderingTask> tasks;

        System system;

        {
            Stream copyStream;
            window.bindBackBuffer(copyStream.get());

            constexpr auto queueSize = 3;

            {
                const auto size = window.size();
                for (auto i = 0; i < queueSize; ++i) {
                    tasks.push(addTask(system,std::make_shared<FrameBuffer>(), size));
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
