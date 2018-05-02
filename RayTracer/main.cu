#include "kernel.hpp"
#include <cstdio>
#include <IO/Model.hpp>
#include <Core/Environment.hpp>
#include <RayTracer/Scene.hpp>
#include <Core/Constant.hpp>

using namespace std::chrono_literals;

struct App final :Uncopyable {
private:
    PinholeCamera mCamera;
    std::unique_ptr<BvhForTriangle> mBvh;
    std::unique_ptr<Constant<BvhForTriangleRef>> mBvhRef;
    MemorySpan<LightWrapperLi> mLight;
    std::unique_ptr<SceneDesc> mScene;
public:
    void run() {
        auto&& env = Environment::get();
        env.init();
         try {
            {
                Stream resLoader;
                auto buffer = std::make_unique<CommandBuffer>();
                StaticMesh model("Res/dragon.obj");
                mBvh = std::make_unique<BvhForTriangle>(model, 32U, resLoader);
                mBvhRef = std::make_unique<Constant<BvhForTriangleRef>>();
                mBvhRef->set(mBvh->getRef(), resLoader);
                mLight = makeLightWrapperLi<PointLight>(*buffer, Point{ 0.0f,10.0f,0.0f }, Spectrum{ 1.0f });
                std::vector<Primitive> primitives;
                primitives.emplace_back(Transform{}, mBvhRef->get(), nullptr);
                std::vector<LightWrapperLi*> li;
                std::vector<LightWrapperLe*> le;
                li.emplace_back(mLight.begin());
                mScene = std::make_unique<SceneDesc>(primitives, li, le);
                env.get().submit(std::move(buffer)).sync();
                resLoader.sync();
            }

            env.uninit();
        }
        catch (const std::exception& e) {
            puts("Catched an error:");
            puts(e.what());
            system("pause");
        }
    }
};

int main() {
    App app;
    app.run();
    return 0;
}
