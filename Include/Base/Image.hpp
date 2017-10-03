#pragma once
#include "Common.hpp"
#include "RenderTarget.hpp"
#include "Sampler.hpp"
#include <string>

template<typename T>
class Image final {
private:
    DataViewer<T> mData;
    size_t mWidth, mHeight;
public:
    Image(size_t width,size_t height):mData(allocBuffer<T>(width*height)),
        mWidth(width), mHeight(height) {}
    Image(void* data,size_t width,size_t height):Image(width,height) {
        auto size = width*height * sizeof(T);
        cudaMemcpy(mData.begin(),data,size,cudaMemcpyDefault);
    }
    auto toTexture() {
        return Sampler<T>(mData.begin(), { mWidth,mHeight });
    }
    auto toRenderTarget() {
        return RenderTarget<T>(mData.begin(), { mWidth,mHeight });
    }
    auto getPtr() const {
        return mData.begin();
    }
};

Image<RGBA> loadRGBA(const std::string& path);
Image<RGB> loadRGB(const std::string& path);
