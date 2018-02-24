#pragma once
#include <Interaction/D3D11.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <ScanLineRenderer/DepthBuffer.hpp>
#include <IO/Model.hpp>
#include <PBR/BRDF.hpp>
#include <PBR/PhotorealisticRendering.hpp>
#include <ScanLineRenderer/RenderingCache.hpp>
#include <ScanLineRenderer/TriangleRasterizer.hpp>

using VI = StaticMesh::Vertex;

enum OutInfo {
    Pos,
    Normal,
    Tangent,
    TexCoord
};

using OI = Args<VAR(Pos, vec3), VAR(Normal, vec3),VAR(Tangent, vec3)>;

struct FrameBufferGPU final {
    BuiltinRenderTargetGPU<RGBA> color;
    DepthBufferGPU<unsigned int> depth;
    uvec2 fsize;
    CUDAINLINE uvec2 size() const {
        return fsize;
    }
};

class ImageResourceInstance final : public ResourceInstance {
private:
    D3D11Image& mImage;
    std::shared_ptr<BuiltinRenderTarget<RGBA8>> mTarget;
    cudaStream_t mStream;
public:
    explicit ImageResourceInstance(D3D11Image& image)
        :mImage(image), mStream(nullptr) {}

    void getRes(void* ptr, const cudaStream_t stream) override {
        if (!mTarget) {
            mTarget = std::make_shared<BuiltinRenderTarget<RGBA8>>(mImage.bind(stream), mImage.size());
            mStream = stream;
        }
        *reinterpret_cast<BuiltinRenderTargetGPU<RGBA8>*>(ptr) = mTarget->toTarget();
    }

    ~ImageResourceInstance() {
        if (mTarget) {
            mTarget.reset();
            mImage.unbind(mStream);
        }
    }
};

class ImageResource final : public Resource<BuiltinRenderTargetGPU<RGBA8>> {
private:
    D3D11Image& mImage;
public:
    ImageResource(ResourceManager& manager, D3D11Image& image)
        : Resource(manager), mImage(image) {}

    ~ImageResource() {
        addInstance(std::make_unique<ImageResourceInstance>(mImage));
    }
};

struct FrameBufferCPU final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<DepthBuffer<unsigned int>> depthBuffer;
    std::unique_ptr<BuiltinRenderTarget<RGBA>> colorRT;
    D3D11Image image;
    uvec2 size;
    FrameBufferGPU data;

    void resize(const uvec2 nsiz) {
        if (size == nsiz)return;
        size = nsiz;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(size,cudaArraySurfaceLoadStore);
        colorRT = std::make_unique<BuiltinRenderTarget<RGBA>>(*colorBuffer);
        depthBuffer = std::make_unique<DepthBuffer<unsigned int>>(size);
        image.resize(size);
        data.color = colorRT->toTarget();
        data.depth = depthBuffer->toBuffer();
        data.fsize = size;
    }

    MemoryRef<FrameBufferGPU> getData(CommandBuffer& buffer) const {
        auto dataGPU = buffer.allocConstant<FrameBufferGPU>();
        buffer.memcpy(dataGPU, [buf=data](auto call) {
            call(&buf);
        });
        return dataGPU;
    }
};

struct Uniform final {
    ALIGN vec2 mul;
    ALIGN mat4 M;
    ALIGN mat4 Msky;
    ALIGN mat4 V;
    ALIGN mat4 invV;
    ALIGN mat3 normalInvV;
    ALIGN mat3 normalMat;
    ALIGN vec3 cp;
    ALIGN vec3 lp;
    ALIGN vec3 lc;
    ALIGN float r2;
    ALIGN DisneyBRDFArg arg;
    ALIGN BuiltinSamplerGPU<RGBA> sampler;
    ALIGN RC8::BlockGPU cache;
};

struct PostUniform final {
    ALIGN FrameBufferGPU in;
    ALIGN float* lum;
    ALIGN std::pair<float, unsigned int>* sum;

    PostUniform(const FrameBufferGPU buf, float* clum, std::pair<float, unsigned int>* cnt)
        : in(buf), lum(clum), sum(cnt) {}
};

void kernel(const StaticMesh& model, TriangleRenderingHistory& mh,
            const StaticMesh& skybox, TriangleRenderingHistory& sh,
            const DataViewer<vec4>& spheres,
            const MemoryRef<Uniform>& uniform, FrameBufferCPU& fbo, float* lum,
            Camera::RasterPosConverter converter, CommandBuffer& buffer);
