#include <IO/Model.hpp>
#include <Core/Environment.hpp>
#include <RayTracer/Scene.hpp>
#include <Core/Constant.hpp>
#include <Camera/PinholeCamera.hpp>
#include <RayTracer/BVH.hpp>
#include <RayTracer/Integrators/Whitted.hpp>
#include <Light/LightWrapper.hpp>
#include <Light/DeltaPositionLight.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <RayTracer/RenderingAPI.hpp>
#include <Core/CompileBegin.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Core/CompileEnd.hpp>
#include <RayTracer/Film.hpp>
#include <Camera/RayGeneratorWrapper.hpp>
#include <IO/Image.hpp>
#include <Material/MaterialWrapper.hpp>

using namespace std::chrono_literals;

struct App final : Uncopyable {
private:
    PinholeCamera mCamera;
    std::unique_ptr<BvhForTriangle> mBvh;
    std::unique_ptr<Constant<BvhForTriangleRef>> mBvhRef;
    MemorySpan<LightWrapper> mLight;
    std::unique_ptr<SceneDesc> mScene;
    std::unique_ptr<WhittedIntegrator> mIntegrator;
    MemorySpan<MaterialWrapper> mMaterial;
public:
    void run() {
        auto&& env = Environment::get();
        env.init();
        {
            Stream resLoader;
            StaticMesh model("Res/cube.obj");
            mBvh = std::make_unique<BvhForTriangle>(model, 32U, resLoader);
            mBvhRef = std::make_unique<Constant<BvhForTriangleRef>>();
            mBvhRef->set(mBvh->getRef(), resLoader);
            mLight = makeLightWrapper<PointLight>(resLoader, Point{ 0.0f, 10.0f, 0.0f }, Spectrum{ 1.0f });
            mMaterial = MemorySpan<MaterialWrapper>(1);
            const TextureMapping2DWrapper mapping{ UVMapping{} };
            const TextureSampler2DSpectrumWrapper samplerS{ ConstantSampler2DSpectrum{Spectrum{0.5f}} };
            const TextureSampler2DFloatWrapper samplerF{ ConstantSampler2DFloat{0.0f} };
            const Texture2DSpectrum textureS{ mapping,samplerS };
            const Texture2DFloat textureF{ mapping ,samplerF };
            MaterialWrapper material{ Plastic{textureS,textureS,textureF} };
            checkError(cudaMemcpyAsync(mMaterial.begin(), &material, sizeof(MaterialWrapper),
                cudaMemcpyHostToDevice, resLoader.get()));
            std::vector<Primitive> primitives;
            primitives.emplace_back(Transform{}, mBvhRef->get(), mMaterial.begin());
            std::vector<LightWrapper*> lights;
            lights.emplace_back(mLight.begin());
            mScene = std::make_unique<SceneDesc>(primitives, lights);
            resLoader.sync();
        }
        SequenceGenerator2DWrapper sequenceGenerator{Halton2D{}};
        const SampleWeightLUT lut(256U, FilterWrapper{TriangleFilter{}});
        const uvec2 imageSize{2U, 2U};
        mIntegrator = std::make_unique<WhittedIntegrator>(sequenceGenerator, 1U, 1U);
        auto res = renderFrame(*mIntegrator, *mScene,
            Transform(glm::lookAt(Vector{}, Vector{0.0f, 0.0f, -1.0f}, Vector{0.0f, 1.0f, 0.0f})),
            RayGeneratorWrapper(mCamera.getRayGenerator(imageSize)), lut, imageSize);
        PinnedBuffer<Spectrum> pixel(res.size());
        cudaMemcpy(pixel.begin(), res.begin(), sizeof(Spectrum) * res.size(), cudaMemcpyDeviceToHost);
        std::vector<float> pixelFloat(pixel.size()*3);
        for (size_t i = 0; i < pixel.size(); ++i) {
            const auto col = pixel[i].toRGB();
            pixelFloat[i * 3] = col.r;
            pixelFloat[i * 3 + 1] = col.g;
            pixelFloat[i * 3 + 2] = col.b;
        }
        saveHdr("output.hdr", pixelFloat.data(), imageSize);
        env.uninit();
    }
};

int main() {
    App app;
    app.run();
    return 0;
}
