#include <Base/Builtin.hpp>
#include <stb/stb_image.h>

std::shared_ptr<BuiltinSampler<RGBA>> builtinLoadRGBA(const std::string& path, 
    cudaTextureAddressMode am, vec4 borderColor,cudaTextureFilterMode fm,
    unsigned int maxAnisotropy,bool sRGB) {
    stbi_set_flip_vertically_on_load(true);
    int x, y, channel;
    auto p = stbi_loadf(path.c_str(), &x, &y, &channel, STBI_rgb_alpha);
    auto res =std::make_shared<BuiltinSampler<RGBA>>
        (x, y, p, am, borderColor, fm, maxAnisotropy, sRGB);
    stbi_image_free(p);
    return res;
}
