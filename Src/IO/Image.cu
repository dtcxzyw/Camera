#include <IO/Image.hpp>
#include <Core/CompileBegin.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <Core/CompileEnd.hpp>

struct ImageDeleter final {
    void operator()(float* ptr) const {
        stbi_image_free(ptr);
    }
};

using ImageHolder = std::unique_ptr<float, ImageDeleter>;

std::shared_ptr<BuiltinArray<RGBA>> loadRGBA(const std::string& path, Stream& stream) {
    stbi_set_flip_vertically_on_load(true);
    int w, h, channel;
    const ImageHolder image(stbi_loadf(path.c_str(), &w, &h, &channel, STBI_rgb_alpha));
    if (!image)throw std::runtime_error(stbi_failure_reason());
    auto res = std::make_shared<BuiltinArray<RGBA>>(uvec2(w, h));
    checkError(cudaMemcpyToArrayAsync(res->get(), 0, 0, image.get(), w * h * sizeof(RGBA)
        , cudaMemcpyHostToDevice, stream.get()));
    stream.sync();
    return res;
}

std::shared_ptr<BuiltinMipmapedArray<RGBA>> loadMipmapedRGBA(const std::string& path, Stream& stream) {
    const auto src = loadRGBA(path, stream);
    return std::make_shared<BuiltinMipmapedArray<RGBA>>(*src, stream);
}

std::shared_ptr<BuiltinCubeMap<RGBA>> loadCubeMap(const std::function<std::string(size_t id)>& path, Stream& stream) {
    stbi_set_flip_vertically_on_load(false);
    std::shared_ptr<BuiltinCubeMap<RGBA>> res;
    std::vector<ImageHolder> freeList;
    for (size_t i = 0; i < 6; ++i) {
        auto rpath = path(i);
        int w, h, channel;
        ImageHolder image(stbi_loadf(rpath.c_str(), &w, &h, &channel, STBI_rgb_alpha));
        if (!image)throw std::runtime_error(stbi_failure_reason());
        if (!res)res = std::make_shared<BuiltinCubeMap<RGBA>>(w);
        cudaMemcpy3DParms parm;
        memset(&parm, 0, sizeof(parm));
        parm.kind = cudaMemcpyHostToDevice;
        parm.extent = make_cudaExtent(w, w, 1);
        parm.srcPtr = make_cudaPitchedPtr(image.get(), sizeof(RGBA) * w, w, h);
        parm.dstArray = res->get();
        parm.dstPos = make_cudaPos(0, 0, i);
        checkError(cudaMemcpy3DAsync(&parm, stream.get()));
        freeList.emplace_back(std::move(image));
    }
    stream.sync();
    return res;
}

std::pair<std::vector<float>, uvec2> loadDistribution2D(const std::string& path) {
    stbi_set_flip_vertically_on_load(true);
    int w, h, channel;
    const ImageHolder image(stbi_loadf(path.c_str(), &w, &h, &channel, STBI_grey));
    if (!image)throw std::runtime_error(stbi_failure_reason());
    return std::make_pair(std::vector<float>{image.get(), image.get() + w * h}, uvec2{w, h});
}

void saveHdr(const std::string& path, const float* pixel, const uvec2 size) {
    const auto res = stbi_write_hdr(path.c_str(), size.x, size.y, 3, pixel);
    if (res == 0)throw std::runtime_error("Failed to save.");
}
