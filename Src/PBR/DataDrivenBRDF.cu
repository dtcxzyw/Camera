#include <PBR/DataDrivenBRDF.hpp>
#include <fstream>

BRDFSampler::BRDFSampler(const vec3* ReadOnly data):mData(data) {}

MERLBRDFData::MERLBRDFData(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::exception("Failed to load the BRDF database.");
    int dims[3];
    in.read(reinterpret_cast<char*>(dims),sizeof(dims));
    auto rsiz = dims[0] * dims[1] * dims[2];
    if (rsiz != size)
        throw std::exception("Failed to load the BRDF database.");
    std::vector<double> brdf(3*rsiz);
    in.read(reinterpret_cast<char*>(brdf.data()),brdf.size()*sizeof(double));
    mData = allocBuffer<vec3>(rsiz);
    for (int i = 0; i < mData.size(); ++i) {
        mData[i].r = static_cast<float>(brdf[i])*rfac;
        mData[i].g = static_cast<float>(brdf[i+size])*gfac;
        mData[i].b = static_cast<float>(brdf[i + size * 2])*bfac;
        mData[i] = clamp(mData[i],0.0f,1.0f);
    }
}

BRDFSampler MERLBRDFData::toSampler() const {
    return { mData.begin() };
}
