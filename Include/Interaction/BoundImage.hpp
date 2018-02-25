#pragma once
#include <Base/DispatchSystem.hpp>
#include <Base/Builtin.hpp>

class BoundImage : Uncopyable {
protected:
    uvec2 mSize;
    cudaGraphicsResource_t mRes;
    virtual void reset() =0;
    void destoryRes();
public:
    BoundImage();
    virtual ~BoundImage() = default;
    uvec2 size() const;
    void resize(uvec2 size);
    virtual cudaArray_t bind(cudaStream_t stream);
    virtual void unbind(cudaStream_t stream);
};

class ImageResourceInstance final : public ResourceInstance {
private:
    BoundImage& mImage;
    std::shared_ptr<BuiltinRenderTarget<RGBA8>> mTarget;
    cudaStream_t mStream;
public:
    explicit ImageResourceInstance(BoundImage& image);
    void getRes(void* ptr, cudaStream_t stream) override;
    ~ImageResourceInstance();
};

class ImageResource final : public Resource<BuiltinRenderTargetGPU<RGBA8>> {
private:
    BoundImage& mImage;
public:
    ImageResource(ResourceManager& manager, BoundImage& image);
    ~ImageResource();
};
