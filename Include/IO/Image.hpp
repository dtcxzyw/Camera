#pragma once
#include <Core/Builtin.hpp>
#include <functional>

std::shared_ptr<BuiltinArray<RGBA>> loadRGBA(const std::string& path, Stream& stream);
std::shared_ptr<BuiltinMipmapedArray<RGBA>> loadMipmapedRGBA(const std::string& path,
    Stream& stream);
std::shared_ptr<BuiltinCubeMap<RGBA>> loadCubeMap(const std::function<std::string(size_t id)>& path,
    Stream& stream);
std::pair<std::vector<float>, uvec2> loadDistribution2D(const std::string& path);
void saveHdr(const std::string& path, const float* pixel, uvec2 size);
