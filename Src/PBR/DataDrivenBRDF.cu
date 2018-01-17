#include <PBR/DataDrivenBRDF.hpp>
#include <fstream>
#include <vector>

BRDFSampler::BRDFSampler(ReadOnlyCache(vec3) data):mData(data) {}

MERLBRDFData::MERLBRDFData(const std::string& path, Stream& loader) {
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
    std::vector<vec3> data(rsiz);
    for (int i = 0; i < mData.size(); ++i) {
        data[i].r = static_cast<float>(brdf[i])*rfac;
        data[i].g = static_cast<float>(brdf[i+size])*gfac;
        data[i].b = static_cast<float>(brdf[i + size * 2])*bfac;
    }
    checkError(cudaMemcpyAsync(mData.begin(), data.data(), sizeof(vec3)*data.size(),
        cudaMemcpyHostToDevice,loader.getID()));
    loader.sync();
}

BRDFSampler MERLBRDFData::toSampler() const {
    return { mData.begin() };
}
