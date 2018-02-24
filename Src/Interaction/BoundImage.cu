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

cudaArray_t BoundImage::bind(const cudaStream_t stream) {
    checkError(cudaGraphicsMapResources(1, &mRes, stream));
    cudaArray_t data;
    checkError(cudaGraphicsSubResourceGetMappedArray(&data, mRes, 0, 0));
    return data;
}

void BoundImage::unbind(const cudaStream_t stream) {
    checkError(cudaGraphicsUnmapResources(1, &mRes, stream));
}

