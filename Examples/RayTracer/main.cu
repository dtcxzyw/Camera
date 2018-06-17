#include <IO/Model.hpp>
#include <Core/Environment.hpp>
#include <RayTracer/Scene.hpp>
#include <Core/Constant.hpp>
#include <Camera/PinholeCamera.hpp>
#include <RayTracer/BVH.hpp>
#include <Light/LightWrapper.hpp>
#include <Light/DeltaPositionLight.hpp>
#include <Light/DistantLight.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <RayTracer/RenderingAPI.hpp>
#include <Core/IncludeBegin.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Core/IncludeEnd.hpp>
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
    std::vector<LightWrapper> mLight;
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
            {
                mLight.emplace_back(PointLight(Point{ 3.0f, 3.0f, 3.0f }, Spectrum{ RGB{10.0f, 20.0f, 30.0f} }));
                mLight.emplace_back(PointLight(Point{ -3.0f, 3.0f, 3.0f }, Spectrum{ RGB{30.0f, 20.0f, 10.0f} }));
                const Sphere sphere{ Transform{glm::translate(glm::mat4{},Vector{0.0f,0.5f,0.0f})},0.2f };
                mLight.emplace_back(DiffuseAreaLight{ Spectrum{0.1f},ShapeWrapper{sphere} });
            }
            const TextureMapping2DWrapper mapping{UVMapping{}};
            std::vector<MaterialWrapper> materials;
            {
                const TextureSampler2DSpectrumWrapper samplerS{ConstantSampler2DSpectrum{Spectrum{1.0f}}};
                const TextureSampler2DFloatWrapper samplerF{ConstantSampler2DFloat{0.1f}};
                const Texture2DSpectrum textureS{mapping, samplerS};
                const Texture2DFloat textureF{mapping, samplerF};
                materials.emplace_back(Subtrate{textureS, textureS, textureF, textureF});
            }
            {
                const TextureSampler2DSpectrumWrapper samplerR{ConstantSampler2DSpectrum{Spectrum{0.2f}}};
                const TextureSampler2DSpectrumWrapper samplerT{ConstantSampler2DSpectrum{Spectrum{0.8f}}};
                const TextureSampler2DFloatWrapper index{ConstantSampler2DFloat{1.01f}};
                const TextureSampler2DFloatWrapper roughness{ConstantSampler2DFloat{0.0f}};
                const Texture2DSpectrum textureR{mapping, samplerR};
                const Texture2DSpectrum textureT{mapping, samplerT};
                const Texture2DFloat indexT{mapping, index};
                const Texture2DFloat roughnessT{mapping, roughness};
                materials.emplace_back(Glass{textureR, textureT, indexT, roughnessT, roughnessT});
            }
            {
                const TextureSampler2DSpectrumWrapper samplerS{ConstantSampler2DSpectrum{Spectrum{1.0f}}};
                const TextureSampler2DFloatWrapper samplerF{ConstantSampler2DFloat{0.01f}};
                const Texture2DSpectrum textureS{mapping, samplerS};
                const Texture2DFloat textureF{mapping, samplerF};
                materials.emplace_back(Metal{textureS, textureS, textureF, textureF});
            }
            mMaterial = upload(materials);

            std::vector<Primitive> primitives;
            const auto sphereMat = glm::translate(glm::mat4{}, {-0.25f, 0.2f, 2.0f}) * glm::scale(glm::mat4{},
                Vector(1e-3f));
            addModel(resLoader, primitives, sphereMat, "Res/sphere.obj", mMaterial.begin() + 1);
            const auto objectMat = glm::scale(glm::mat4{}, Vector(5.0f));
            addModel(resLoader, primitives, objectMat, "Res/dragon.obj", mMaterial.begin() + 2);
            const auto boxMat = glm::translate(glm::mat4{}, {0.0f, 2.2f, 2.5f}) * glm::scale(glm::mat4{},
                Vector(3.1e-2f));
            addModel(resLoader, primitives, boxMat, "Res/cube.obj", mMaterial.begin());

            const Transform trans(boxMat);
            const Point p0{-100.0f, -100.0f, -100.0f};
            const Point p1{100.0f, 100.0f, 100.0f};
            const auto worldBound = trans(Bounds{ p0, p1 });
            const auto sphere = worldBound.boundingSphere();
            for (auto&& light : mLight)
                light.preprocess(sphere.first, sphere.second);
            mScene = std::make_unique<SceneDesc>(primitives, mLight, worldBound, 32U);

            resLoader.sync();
        }
        SequenceGenerator2DWrapper sequenceGenerator{Halton2D{}};
        const SampleWeightLUT lut(64U, FilterWrapper{TriangleFilter{}});
        const uvec2 imageSize{1920U, 1080U};
        mIntegrator = std::make_unique<PathIntegrator>(sequenceGenerator, 20U, 16U, 256U);

        const Transform toCamera{
            glm::lookAt(Vector{0.0f, 0.0f, 3.0f}, Vector{0.0f, 0.0f, 0.0f}, Vector{0.0f, 1.0f, 0.0f})
        };

        mCamera.lensRadius = 2.0f;
        mCamera.focalDistance = 3.0f;
        mCamera.fov = 55.0f;

        const auto beg = Clock::now();

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
            auto col = pixel[i].toRGB();
            if (!isfinite(pixel[i].y())) {
                valid = false;
                col = { 0.0f,1.0f,0.0f };
            }
            pixelFloat[i * 3] = col.r;
            pixelFloat[i * 3 + 1] = col.g;
            pixelFloat[i * 3 + 2] = col.b;
        }
        saveHdr("output.hdr", pixelFloat.data(), imageSize);
        if (!valid)printf("The image is invalid.\n");
        system("pause");
        mScene.reset();
        env.uninit();
    }
};

int main() {
    App app;
    app.run();
    return 0;
}
