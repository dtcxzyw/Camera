#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <regex>
using namespace std::experimental::filesystem;

bool scan(const path& path, const std::string& base, std::vector<std::string>& out) {
    std::cout << "Scaning " << path << std::endl;
    std::ifstream in(path.c_str(), std::ios::binary);
    in.seekg(0, std::ios::end);
    const size_t len = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> data(len + 10);
    in.read(data.data(), len);
    std::string str(data.data());
    const std::regex pattern(R"(class\s([A-Z][A-Za-z0-9]+)\s*(final)?\s*:\s*(public)?\s*)" + base);
    size_t count = 0;
    for (auto it = std::sregex_iterator(str.begin(), str.end(), pattern); it != std::sregex_iterator{}; ++it) {
        const auto className = it->str(1);
        std::cout << "find derived class " << className << std::endl;
        out.emplace_back(className);
        ++count;
    }
    return count;
}

struct ApiDesc {
    std::string returnType;
    std::string funcName;
    std::string argDesc;
    std::string arg;
    bool isConst;

    ApiDesc(const std::string& returnType, const std::string& funcName, const std::string& argDesc,
        const std::string& arg, bool isConst = true) : returnType(returnType), funcName(funcName), argDesc(argDesc),
        arg(arg), isConst(isConst) {}
};

constexpr auto line = "------------------------------------------------------";

void generateWrapper(const std::string& dir, const std::string& base, const std::string& type,
    const std::vector<ApiDesc>& apiDesc, const std::vector<std::string>& forceInline = {},
    const bool constructOnHost = true) {
    std::cout << line << std::endl;
    std::cout << "Generating " << type << "Wrapper" << std::endl;
    const auto path = current_path().parent_path().parent_path() / "Include" / dir;
    std::vector<std::string> includes;
    std::vector<std::string> derived;
    for (auto&& file : directory_iterator(path))
        if (file.status().type() == file_type::regular) {
            if (scan(file, base, derived))
                includes.emplace_back(file.path().filename().string());
        }

    const auto wrapperName = type + "Wrapper";

    std::ofstream res(path / (wrapperName + ".hpp"));
    const std::string tab1 = "    ";
    const auto tab2 = tab1 + tab1;
    const auto tab3 = tab2 + tab1;
    const auto tab4 = tab2 + tab2;

    res << "#pragma once" << std::endl;
    for (auto&& file : forceInline)
        res << "#include <" << file << ">" << std::endl;
    for (auto&& file : includes)
        res << "#include <" << dir << "/" << file << ">" << std::endl;
    res << std::endl;
    res << "class " << wrapperName << " final {" << std::endl;
    res << "private:" << std::endl;

    const auto enumName = type + "ClassType";

    res << tab1 << "enum class " << enumName << " {" << std::endl;
    for (size_t i = 0; i < derived.size(); ++i) {
        res << tab2 << derived[i] << " = " << i;
        if (i < derived.size() - 1)res << ",";
        res << std::endl;
    }
    res << tab1 << "};" << std::endl << std::endl;

    res << tab1 << "union {" << std::endl;
    res << tab2 << "unsigned char unused{};" << std::endl;
    for (size_t i = 0; i < derived.size(); ++i)
        res << tab2 << derived[i] << " " << "data" << derived[i] << ";" << std::endl;
    res << tab1 << "};" << std::endl;

    res << std::endl << tab1 << enumName << " mType;" << std::endl;
    res << "public:" << std::endl;

    const auto construct = constructOnHost ? "" : "DEVICE ";
    res << tab1 << construct << wrapperName << "(): mType(static_cast<" << enumName
        << ">(0xff)) {};" << std::endl << std::endl;
    for (size_t i = 0; i < derived.size(); ++i) {
        res << tab1 << construct << "explicit " << wrapperName << "(const " << derived[i]
            << "& data)" << std::endl;
        res << tab2 << ": data" << derived[i] << "(data), mType(" << enumName << "::"
            << derived[i] << ") {}" << std::endl << std::endl;
    }

    //Copy
    res << tab1 << "BOTH " << wrapperName << "(const " << wrapperName << "& rhs) {"
        << std::endl << tab2 << "memcpy(this, &rhs, sizeof(" << wrapperName << "));"
        << std::endl << tab1 << "}" << std::endl << std::endl;

    res << tab1 << "BOTH " << wrapperName << "& operator=(const " << wrapperName
        << "& rhs) {" << std::endl << tab2 << "memcpy(this, &rhs, sizeof(" << wrapperName << "));"
        << std::endl << tab2 << "return *this;" << std::endl << tab1 << "}" << std::endl << std::endl;
    //API

    for (auto&& api : apiDesc) {
        res << tab1 << "DEVICE " << api.returnType << " " << api.funcName << "(" << api.argDesc
            << ") " << (api.isConst ? "const " : "") << "{" << std::endl;
        res << tab2 << "switch (mType) {" << std::endl;
        for (auto&& name : derived)
            res << tab3 << "case " << enumName << "::" << name << ": return data" << name
                << "." << api.funcName << "(" << api.arg << ");" << std::endl;
        res << tab2 << "}" << std::endl;
        res << tab1 << "}" << std::endl << std::endl;
    }

    res << "};" << std::endl;
    res.close();
}

int main() {
    std::cout << "WrapperGen (built " << __DATE__ << " at " << __TIME__ << ")" << std::endl;

    //BxDF
    generateWrapper("BxDF", "BxDFHelper", "BxDF", {
        {"BxDFType", "getType", "", ""},
        {"float", "pdf", "const Vector& wo, const Vector& wi", "wo, wi"},
        {"Spectrum", "f", "const Vector& wo, const Vector& wi", "wo, wi"},
        {"BxDFSample", "sampleF", "const Vector& wo, const vec2 sample", "wo, sample"},
        {"bool", "match", "const BxDFType pattern", "pattern"}
    }, {}, false);
    //TextureMapping2D
    generateWrapper("Texture", "TextureMapping2D", "TextureMapping2D", {
        {"TextureMapping2DInfo", "map", "const Interaction& interaction", "interaction"}
    });
    //TextureSampler2D
    generateWrapper("Texture", "TextureSampler2DFloatTag", "TextureSampler2DFloat", {
        {"float", "sample", "const TextureMapping2DInfo& info", "info"}
    }, {"Texture/TextureMapping.hpp"});
    generateWrapper("Texture", "TextureSampler2DSpectrumTag", "TextureSampler2DSpectrum", {
        {"Spectrum", "sample", "const TextureMapping2DInfo& info", "info"}
    }, {"Texture/TextureMapping.hpp", "Spectrum/SpectrumConfig.hpp"});
    //TextureSampler3D
    generateWrapper("Texture", "TextureSampler3DFloatTag", "TextureSampler3DFloat", {
        {"float", "sample", "const TextureMapping3DInfo& info", "info"}
    }, {"Texture/TextureMapping.hpp"});
    generateWrapper("Texture", "TextureSampler3DSpectrumTag", "TextureSampler3DSpectrum", {
        {"Spectrum", "sample", "const TextureMapping3DInfo& info", "info"}
    }, {"Texture/TextureMapping.hpp", "Spectrum/SpectrumConfig.hpp"});
    //Material
    generateWrapper("Material", "Material", "Material", {
        {
            "void", "computeScatteringFunctions", "Bsdf& bsdf, const TransportMode mode = TransportMode::Radiance",
            "bsdf, mode"
        }
    });
    //Sampler
    generateWrapper("Sampler", "SequenceGenerator1DTag", "SequenceGenerator1D", {
        {"float", "sample", "const uint32_t index", "index"}
    });
    generateWrapper("Sampler", "SequenceGenerator2DTag", "SequenceGenerator2D", {
        {"vec2", "sample", "const uint32_t index", "index"}
    });
    //Camera
    generateWrapper("Camera", "RayGeneratorTag", "RayGenerator", {
        {"Ray", "sample", "const CameraSample& sample, float& weight", "sample, weight"}
    });
    //Filter
    generateWrapper("Texture", "FilterTag", "Filter", {
        {"float", "evaluate", "const vec2 pos", "pos"}
    });

    std::cout << line << std::endl;
    std::cout << "Done." << std::endl << std::endl;
    std::cin.get();
    return 0;
}
