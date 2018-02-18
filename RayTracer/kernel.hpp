#pragma once
#include <Interaction/OpenGL.hpp>
#include <Base/DispatchSystem.hpp>
#include <Base/DataSet.hpp>
#include <Base/Builtin.hpp>
#include <IO/Model.hpp>
#include <PBR/BRDF.hpp>
#include <PBR/PhotorealisticRendering.hpp>

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
    uvec2 fsize;
    CUDAINLINE uvec2 size() const {
        return fsize;
    }
};

class ImageResourceInstance final : public ResourceInstance {
private:
    Image& mImage;
    std::shared_ptr<BuiltinRenderTarget<RGBA>> mTarget;
    cudaStream_t mStream;
public:
    explicit ImageResourceInstance(Image& image): mImage(image), mStream(nullptr) {}

    void getRes(void* ptr, const cudaStream_t stream) override {
        if (!mTarget) {
            mTarget = std::make_shared<BuiltinRenderTarget<RGBA>>(mImage.bind(stream), mImage.size());
            mStream = stream;
        }
        *reinterpret_cast<BuiltinRenderTargetGPU<RGBA>*>(ptr) = mTarget->toTarget();
    }

    ~ImageResourceInstance() {
        if (mTarget) {
            mTarget.reset();
            mImage.unbind(mStream);
        }
    }
};

class ImageResource final : public Resource<BuiltinRenderTargetGPU<RGBA>> {
private:
    Image& mImage;
public:
    ImageResource(CommandBuffer& buffer, Image& image): Resource(buffer), mImage(image) {}

    ~ImageResource() {
        addInstance(std::make_unique<ImageResourceInstance>(mImage));
    }
};

struct FrameBufferCPU final {
    std::unique_ptr<BuiltinArray<RGBA>> colorBuffer;
    std::unique_ptr<BuiltinRenderTarget<RGBA>> colorRT;
    Image image;
    uvec2 size;
    FrameBufferGPU data;

    void resize(uvec2 nsiz) {
        if (size == nsiz)return;
        size = nsiz;
        colorBuffer = std::make_unique<BuiltinArray<RGBA>>(size,cudaArraySurfaceLoadStore);
        colorRT = std::make_unique<BuiltinRenderTarget<RGBA>>(*colorBuffer);
        image.resize(size);
        data.color = colorRT->toTarget();
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

struct PostUniform final {
    ALIGN FrameBufferGPU in;
    ALIGN float* lum;
    ALIGN std::pair<float, unsigned int>* sum;

    PostUniform(const FrameBufferGPU buf, float* clum, std::pair<float, unsigned int>* cnt)
        : in(buf), lum(clum), sum(cnt) {}
};

