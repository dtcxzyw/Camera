#include <IO/Model.hpp>
#include <Core/Environment.hpp>
#include <RayTracer/Scene.hpp>
#include <Core/Constant.hpp>
#include <Camera/PinholeCamera.hpp>
#include <RayTracer/BVH.hpp>
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
#include <RayTracer/Integrators/Path.hpp>
#include <RayTracer/Integrators/Whitted.hpp>

using namespace std::chrono_literals;

struct App final : Uncopyable {
private:
    PinholeCamera mCamera;
    std::vector<std::unique_ptr<BvhForTriangle>> mBvh;
    std::vector<std::unique_ptr<Constant<BvhForTriangleRef>>> mBvhRef;
    std::vector<MemorySpan<LightWrapper>> mLight;
    std::unique_ptr<SceneDesc> mScene;
    std::unique_ptr<PathIntegrator> mIntegrator;
    MemorySpan<MaterialWrapper> mMaterial;
public:
    void addModel(Stream& resLoader, std::vector<Primitive>& primitives,
        const glm::mat4& trans, const std::string& path, MaterialWrapper* material) {
        StaticMesh mesh(path);
        mBvh.emplace_back(std::make_unique<BvhForTriangle>(mesh, 32U, resLoader));
        mBvhRef.emplace_back(std::make_unique<Constant<BvhForTriangleRef>>());
        mBvhRef.back()->set(mBvh.back()->getRef(), resLoader);
        primitives.emplace_back(Primitive{Transform(trans), mBvhRef.back()->get(), material});
    }

    void run() {
        auto&& env = Environment::get();
        env.init(AppType::Online);
        {
            Stream resLoader;
            mLight.emplace_back(makeLightWrapper<PointLight>(resLoader, Point{3.0f, 3.0f, 3.0f},
                Spectrum{RGB{10.0f, 20.0f, 30.0f}}));
            mLight.emplace_back(makeLightWrapper<PointLight>(resLoader, Point{-3.0f, 3.0f, 3.0f},
                Spectrum{RGB{30.0f, 20.0f, 10.0f}}));
            mMaterial = MemorySpan<MaterialWrapper>(2);
            const TextureMapping2DWrapper mapping{UVMapping{}};
            {
                const TextureSampler2DSpectrumWrapper samplerS{ConstantSampler2DSpectrum{Spectrum{1.0f}}};
                const TextureSampler2DFloatWrapper samplerF{ConstantSampler2DFloat{0.1f}};
                const Texture2DSpectrum textureS{mapping, samplerS};
                const Texture2DFloat textureF{mapping, samplerF};
                MaterialWrapper plastic{Plastic{textureS, textureS, textureF}};
                checkError(cudaMemcpyAsync(mMaterial.begin(), &plastic, sizeof(MaterialWrapper),
                    cudaMemcpyHostToDevice, resLoader.get()));
            }
            {
                const TextureSampler2DSpectrumWrapper samplerR{ConstantSampler2DSpectrum{Spectrum{0.0f}}};
                const TextureSampler2DSpectrumWrapper samplerT{ConstantSampler2DSpectrum{Spectrum{1.0f}}};
                const TextureSampler2DFloatWrapper index{ConstantSampler2DFloat{1.5f}};
                const TextureSampler2DFloatWrapper roughness{ConstantSampler2DFloat{0.0f}};
                const Texture2DSpectrum textureR{mapping, samplerR};
                const Texture2DSpectrum textureT{mapping, samplerT};
                const Texture2DFloat indexT{mapping, index};
                const Texture2DFloat roughnessT{mapping, roughness};
                MaterialWrapper glass{Glass{textureR, textureT, indexT, roughnessT, roughnessT}};
                checkError(cudaMemcpyAsync(mMaterial.begin() + 1, &glass, sizeof(MaterialWrapper),
                    cudaMemcpyHostToDevice, resLoader.get()));
            }

            std::vector<Primitive> primitives;
            const auto cubeMat = glm::scale(glm::rotate(
                glm::translate(glm::mat4{}, {0.0f, 0.0f, 1.0f}), 45.0f, Vector(1.0f)), Vector(1e-3f));
            addModel(resLoader, primitives, cubeMat, "Res/cube.obj", mMaterial.begin() + 1);
            const auto objectMat = glm::scale(glm::mat4{}, Vector(5.0f));
            addModel(resLoader, primitives, objectMat, "Res/dragon.obj", mMaterial.begin());
            std::vector<LightWrapper*> lights;
            for (auto&& light : mLight)
                lights.emplace_back(light.begin());
            mScene = std::make_unique<SceneDesc>(primitives, lights);
            resLoader.sync();
        }
        SequenceGenerator2DWrapper sequenceGenerator{Halton2D{}};
        const SampleWeightLUT lut(64U, FilterWrapper{TriangleFilter{}});
        const uvec2 imageSize{1920U, 1080U};
        mIntegrator = std::make_unique<PathIntegrator>(sequenceGenerator, 10U, 1024U, 256U);
        const auto beg = Clock::now();
        const Transform toCamera{
            glm::lookAt(Vector{0.0f, 0.0f, 2.0f}, Vector{0.0f, 0.0f, 0.0f}, Vector{0.0f, 1.0f, 0.0f})
        };
        mCamera.near = 1.0f;
        auto res = renderFrame(*mIntegrator, *mScene, inverse(toCamera),
            RayGeneratorWrapper(mCamera.getRayGenerator(imageSize)), lut, imageSize, 32U);
        const auto end = Clock::now();
        const auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
        printf("%.3lf ms\n", t * 1e-3);
        PinnedBuffer<Spectrum> pixel(res.size());
        cudaMemcpy(pixel.begin(), res.begin(), sizeof(Spectrum) * res.size(), cudaMemcpyDeviceToHost);
        std::vector<float> pixelFloat(pixel.size() * 3);
        auto valid = true;
        for (size_t i = 0; i < pixel.size(); ++i) {
            const auto col = pixel[i].toRGB();
            pixelFloat[i * 3] = col.r;
            pixelFloat[i * 3 + 1] = col.g;
            pixelFloat[i * 3 + 2] = col.b;
            valid &= (isfinite(pixel[i].lum()));
        }
        saveHdr("output.hdr", pixelFloat.data(), imageSize);
        if (!valid)printf("The image is invalid.");
        system("pause");
        env.uninit();
    }
};

int main() {
    App app;
    app.run();
    return 0;
}
