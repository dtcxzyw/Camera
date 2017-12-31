#include <IO/Image.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

std::shared_ptr<BuiltinArray<RGBA>> loadRGBA(const std::string& path,Stream& stream) {
    stbi_set_flip_vertically_on_load(true);
    int w, h, channel;
    auto p = stbi_loadf(path.c_str(), &w, &h, &channel, STBI_rgb_alpha);
    if (p == nullptr)
        throw std::exception(stbi_failure_reason());
    auto res = std::make_shared<BuiltinArray<RGBA>>(w,h);
    checkError(cudaMemcpyToArrayAsync(res->get(), 0, 0, p, w*h * sizeof(RGBA)
        , cudaMemcpyHostToDevice,stream.getID()));
    stream.sync();
    stbi_image_free(p);
    return res;
}

std::shared_ptr<BuiltinMipmapedArray<RGBA>> loadMipmapedRGBA(const std::string & path, Stream & stream) {
    stbi_set_flip_vertically_on_load(true);
    int w, h, channel;
    auto p = stbi_loadf(path.c_str(), &w, &h, &channel, STBI_rgb_alpha);
    if (p == nullptr)
        throw std::exception(stbi_failure_reason());
    auto res = std::make_shared<BuiltinMipmapedArray<RGBA>>(p,w, h,stream);
    stbi_image_free(p);
    return res;
}

std::shared_ptr<BuiltinCubeMap<RGBA>> loadCubeMap(const std::function<std::string(size_t id)>& path, Stream & stream) {
    stbi_set_flip_vertically_on_load(true);
    std::shared_ptr<BuiltinCubeMap<RGBA>> res;
    for (size_t i = 0; i < 6; ++i) {
        auto rpath = path(i);
        int w, h, channel;
        auto p = stbi_loadf(rpath.c_str(), &w, &h, &channel, STBI_rgb_alpha);
        if (p == nullptr)
            throw std::exception(stbi_failure_reason());
        if (!res)res = std::make_shared<BuiltinCubeMap<RGBA>>(w);
        cudaMemcpy3DParms parm;
        parm.kind = cudaMemcpyHostToDevice;
        parm.extent = { w,w,1 };
        parm.srcPtr =make_cudaPitchedPtr(p,sizeof(RGBA)*w,w,h);
        parm.dstArray = res->get();
        parm.dstPos = make_cudaPos(0,0,i);
        checkError(cudaMemcpy3DAsync(&parm,stream.getID()));
        stream.sync();
        stbi_image_free(p);
    }
    return res;
}
