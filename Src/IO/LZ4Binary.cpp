#include <IO/LZ4Binary.hpp>
#include <Core/CompileBegin.hpp>
#include <lz4.h>
#include <Core/CompileEnd.hpp>
#include <fstream>

std::vector<char> loadLZ4(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (in) {
        in.seekg(0, std::ios::end);
        const auto siz = static_cast<uint64_t>(in.tellg()) - sizeof(uint64_t);
        in.seekg(0);
        std::vector<char> data(siz);
        uint64_t srcSize;
        in.read(reinterpret_cast<char*>(&srcSize), sizeof(uint64_t));
        in.read(data.data(), siz);
        std::vector<char> res(srcSize);
        LZ4_decompress_fast(data.data(), res.data(), static_cast<int>(srcSize));
        return res;
    }
    return {};
}

void saveLZ4(const std::string& path, const std::vector<char>& data) {
    std::ofstream out(path, std::ios::binary);
    if (out) {
        const uint64_t srcSize = data.size();
        out.write(reinterpret_cast<const char*>(&srcSize), sizeof(uint64_t));
        std::vector<char> res(LZ4_compressBound(static_cast<int>(data.size())));
        const auto dstSize = LZ4_compress_default(data.data(), res.data(),
            static_cast<int>(srcSize), static_cast<int>(res.size()));
        out.write(res.data(), dstSize);
    }
    else throw std::runtime_error("Failed to save LZ4 binary.");
}
