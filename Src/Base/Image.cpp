#include <Base/Image.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

void init() {
    stbi_set_flip_vertically_on_load(true);
}

Image<RGBA> loadRGBA(const std::string & path) {
    init();
    int x, y, channel;
    auto p=stbi_loadf(path.c_str(), &x, &y, &channel, STBI_rgb_alpha);
    Image<RGBA> res(p, x, y);
    stbi_image_free(p);
    return res;
}

Image<RGB> loadRGB(const std::string & path) {
    init();
    int x, y, channel;
    auto p = stbi_loadf(path.c_str(), &x, &y, &channel, STBI_rgb);
    Image<RGB> res(p, x, y);
    stbi_image_free(p);
    return res;
}
