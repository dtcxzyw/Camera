#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <regex>
using namespace std::experimental::filesystem;

bool scan(const path& path,std::vector<std::string>& out) {
    std::cout << "Scaning " << path << std::endl;
    std::ifstream in(path.c_str(), std::ios::binary);
    in.seekg(0,std::ios::end);
    const size_t len = in.tellg();
    in.seekg(0,std::ios::beg);
    std::vector<char> data(len + 10);
    in.read(data.data(), len);
    std::string str(data.data());
    const std::regex patten(R"(class\s([A-Z][A-Za-z0-9]+)\s(final)?\s:\spublic\sBxDFHelper)");
    size_t count = 0;
    for (auto it = std::sregex_iterator(str.begin(), str.end(), patten); it != std::sregex_iterator{}; ++it) {
        const auto className = it->str(1);
        std::cout << "find BxDF " << className << std::endl;
        out.emplace_back(className);
        ++count;
    }
    return count;
}

int main(){
    const auto bxdfPath = current_path().parent_path().parent_path()/"Include"/"BxDF";
    std::vector <std::string> includes;
    std::vector <std::string> bxdfs;
    for (auto&& file : directory_iterator(bxdfPath))
        if (file.status().type() == file_type::regular) {
            if(scan(file, bxdfs))
                includes.emplace_back(file.path().filename().string());
        }

    std::ofstream res(bxdfPath / "BxDFWarpper.hpp");
    const std::string tab1 = "    ";
    const auto tab2 = tab1 + tab1;
    const auto tab3 = tab2 + tab1;
    const auto tab4 = tab2 + tab2;

    res << "#pragma once" << std::endl;
    for (auto&& file : includes)
        res << "#include <BxDF/" << file << ">" << std::endl;
    res << std::endl;
    res << "class BxDFWarpper final {" << std::endl;
    res << "private:" << std::endl;

    res << tab1<<"enum class BxDFClassType {"<<std::endl;
    for (size_t i = 0; i < bxdfs.size(); ++i) {
        res << tab2 << bxdfs[i];
        if (i < bxdfs.size() - 1)res << ",";
        res << std::endl;
    }
    res << tab1 << "};" << std::endl << std::endl;

    res << tab1 << "union {" << std::endl;
    res << tab2 << "char unused{};" << std::endl;
    for (size_t i = 0; i < bxdfs.size(); ++i)
        res << tab2 << bxdfs[i] << " " << "bxdf" << bxdfs[i] << ";" << std::endl;
    res << tab1 << "};" << std::endl;

    res << std::endl << tab1 << "BxDFClassType mType;" << std::endl;
    res << "public:" << std::endl;

    res << tab1 << "CUDA BxDFWarpper(): mType(static_cast<BxDFClassType>(15)) {};" << std::endl << std::endl;
    for (size_t i = 0; i < bxdfs.size(); ++i) {
        res << tab1 << "CUDA explicit BxDFWarpper(const " << bxdfs[i] << "& data)" << std::endl;
        res << tab2 << ": bxdf" << bxdfs[i] << "(data), mType(BxDFClassType::" << bxdfs[i] << ") {}"
            << std::endl << std::endl;
    }

    const auto gen=[&](const auto rt,const auto func,const auto arg1,const auto arg2) {
        res << tab1 << "CUDA " << rt << " " << func << "(" << arg1 << ") const {" << std::endl;
        res << tab2 << "switch (mType) {" << std::endl;
        for (auto&& bxdf : bxdfs)
            res << tab3 << "case BxDFClassType::" << bxdf << ": return bxdf" << bxdf 
                << "." << func << "(" << arg2 << ");" << std::endl;
        res << tab2 << "}" << std::endl;
        res << tab1 << "}" << std::endl << std::endl;
    };

    gen("float","pdf","const Vector& wo, const Vector& wi","wo, wi");
    gen("Spectrum","f","const Vector& wo, const Vector& wi","wo, wi");
    gen("BxDFSample","sampleF","const Vector& wo, const vec2 sample","wo, sample");
    gen("bool", "match", "const BxDFType partten", "partten");

    res << "};" << std::endl;
    res.close();
    std::cin.get();
    return 0;
}

