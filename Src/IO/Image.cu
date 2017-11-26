#include <IO/Image.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

std::shared_ptr<BuiltinArray<RGBA>> loadRGBA(const std::string& path,Stream& stream) {
    stbi_set_flip_vertically_on_load(true);
    int w, h, channel;
    auto p = stbi_loadf(path.c_str(), &w, &h, &channel, STBI_rgb_alpha);
    auto res = std::make_shared<BuiltinArray<RGBA>>(w,h);
    checkError(cudaMemcpyToArray(res->get(), 0, 0, p, w*h * sizeof(RGBA)
        , cudaMemcpyHostToDevice));
    stbi_image_free(p);
    return res;
}
