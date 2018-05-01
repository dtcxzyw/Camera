#include "kernel.hpp"
#include <cstdio>
#include <IO/Model.hpp>
#include <RayTracer/BVH.hpp>
#include <Core/Environment.hpp>
#include <Camera/PinholeCamera.hpp>

using namespace std::chrono_literals;

struct App final :Uncopyable {
private:
    PinholeCamera mCamera;
    std::unique_ptr<BvhForTriangle> mBvh;
public:
    void run() {
        auto&& env = Environment::get();
        env.init();

        try {
            {
                Stream resLoader;
                StaticMesh model("Res/dragon.obj");
                mBvh = std::make_unique<BvhForTriangle>(model, 32U, resLoader);

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
