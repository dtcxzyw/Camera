#include <IO/Model.hpp>
#include <Base/CompileBegin.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <Base/CompileEnd.hpp>
#include <vector>
#include <fstream>
#include <filesystem>

template<typename T>
void read(std::ifstream& stream,T* ptr, const size_t size = 1) {
    stream.read(reinterpret_cast<char*>(ptr),sizeof(T)*size);
}

template<typename T>
void write(std::ofstream& stream,const T* ptr, const size_t size=1) {
    stream.write(reinterpret_cast<const char*>(ptr), sizeof(T)*size);
}

void StaticMesh::convertToBinary(const std::string & path) {
    Assimp::Importer importer;
    const auto scene = importer.ReadFile(path, aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType |
        aiProcess_GenSmoothNormals |
        aiProcess_GenUVCoords|
        aiProcess_CalcTangentSpace|
        aiProcess_FixInfacingNormals|
        aiProcess_ImproveCacheLocality
    );

    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE)
        throw std::runtime_error("Failed to load the mesh.");
    const auto mesh=scene->mMeshes[0];
    std::ofstream out(path + ".bin", std::ios::binary|std::ios::out|std::ios::trunc);
    if (!out.is_open())
        throw std::runtime_error("Failed to open the output file.");
    {
        std::vector<Vertex> buf(mesh->mNumVertices);
        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            buf[i].pos = *reinterpret_cast<vec3*>(mesh->mVertices + i);
            buf[i].normal = *reinterpret_cast<vec3*>(mesh->mNormals + i);
            buf[i].uv = *reinterpret_cast<UV*>(mesh->mTextureCoords[0] + i);
            buf[i].tangent = *reinterpret_cast<vec3*>(mesh->mTangents+i);
        }
        const uint64_t vertSize = mesh->mNumVertices;
        write(out,&vertSize);
        write(out, buf.data(), buf.size());
    }
    {
        std::vector<uvec3> buf(mesh->mNumFaces);
        for (uint i = 0; i < mesh->mNumFaces; ++i)
            buf[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
        const uint64_t faceSize = mesh->mNumFaces;
        write(out, &faceSize);
        write(out, buf.data(), buf.size());
    }
    importer.FreeScene();
}

void StaticMesh::loadBinary(const std::string & path, Stream & loader) {
    std::ifstream in(path, std::ios::binary | std::ios::in);
    if (!in.is_open())
        throw std::runtime_error("Failed to open the input file.");
    uint64_t vertSize;
    read(in, &vertSize);
    PinnedBuffer<Vertex> vertHost(vertSize);
    read(in, vertHost.get(), vertSize);
    vert = allocBuffer<Vertex>(vertSize);
    checkError(cudaMemcpyAsync(vert.begin(),vertHost.get(),vertSize*sizeof(Vertex),
        cudaMemcpyHostToDevice,loader.get()));
    uint64_t faceSize;
    read(in, &faceSize);
    PinnedBuffer<uvec3> indexHost(faceSize);
    read(in, indexHost.get(),faceSize);
    index = allocBuffer<uvec3>(faceSize);
    checkError(cudaMemcpyAsync(index.begin(),indexHost.get(),faceSize*sizeof(uvec3),
        cudaMemcpyHostToDevice,loader.get()));
    loader.sync();
}

void StaticMesh::load(const std::string& path, Stream& loader) {
    if (!std::experimental::filesystem::exists(path + ".bin"))
        convertToBinary(path);
    loadBinary(path + ".bin",loader);
}
