#include <Interaction/BoundImage.hpp>

void BoundImage::destoryRes() {
    if (mRes) {
        checkError(cudaGraphicsUnregisterResource(mRes));
        mRes = nullptr;
    }
}

BoundImage::BoundImage(): mRes(nullptr) {}

uvec2 BoundImage::size() const {
    return mSize;
}

void BoundImage::resize(const uvec2 size) {
    if(mSize!=size) {
        destoryRes();
        mSize = size;
        reset();
    }
}

cudaArray_t BoundImage::bind(cudaStream_t stream) {
    checkError(cudaGraphicsMapResources(1, &mRes, stream));
    cudaArray_t data;
    checkError(cudaGraphicsSubResourceGetMappedArray(&data, mRes, 0, 0));
    return data;
}

void BoundImage::unbind(cudaStream_t stream) {
    checkError(cudaGraphicsUnmapResources(1, &mRes, stream));
}

ImageResourceInstance::ImageResourceInstance(BoundImage& image)
    : mImage(image), mStream(nullptr) {}

void ImageResourceInstance::getRes(void* ptr, cudaStream_t stream) {
    if (!mTarget) {
        mTarget = std::make_shared<BuiltinRenderTarget<RGBA8>>(mImage.bind(stream), mImage.size());
        mStream = stream;
    }
    *reinterpret_cast<BuiltinRenderTargetGPU<RGBA8>*>(ptr) = mTarget->toTarget();
}

ImageResourceInstance::~ImageResourceInstance() {
    if (mTarget) {
        mTarget.reset();
        mImage.unbind(mStream);
    }
}

ImageResource::ImageResource(ResourceManager& manager, BoundImage& image) 
    : Resource(manager), mImage(image) {}

ImageResource::~ImageResource() {
    addInstance(std::make_unique<ImageResourceInstance>(mImage));
}
