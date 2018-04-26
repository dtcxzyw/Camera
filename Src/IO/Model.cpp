#include <IO/Model.hpp>
#include <Core/CompileBegin.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <Core/CompileEnd.hpp>
#include <IO/LZ4Binary.hpp>
#include <filesystem>

template<typename T>
void read(const std::vector<char>& stream, uint64_t& offset, T* ptr, const size_t size = 1) {
    const auto rsiz= sizeof(T)*size;
    memcpy(ptr, stream.data() + offset, rsiz);
    offset+=rsiz;
}

template<typename T>
void write(std::vector<char>& stream, const T* ptr, const size_t size = 1) {
    const auto begin = reinterpret_cast<const char*>(ptr);
    stream.insert(stream.end(), begin, begin + sizeof(T)*size);
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
    std::vector<char> data;
    {
        std::vector<Vertex> buf(mesh->mNumVertices);
        for (auto i = 0U; i < mesh->mNumVertices; ++i) {
            buf[i].pos = *reinterpret_cast<Point*>(mesh->mVertices + i);
            buf[i].normal = *reinterpret_cast<Vector*>(mesh->mNormals + i);
            buf[i].uv = *reinterpret_cast<UV*>(mesh->mTextureCoords[0] + i);
            buf[i].tangent = *reinterpret_cast<Vector*>(mesh->mTangents+i);
        }
        const uint64_t vertSize = mesh->mNumVertices;
        write(data,&vertSize);
        write(data, buf.data(), buf.size());
    }
    {
        std::vector<uvec3> buf(mesh->mNumFaces);
        for (auto i = 0U; i < mesh->mNumFaces; ++i)
            buf[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
        const uint64_t faceSize = mesh->mNumFaces;
        write(data, &faceSize);
        write(data, buf.data(), buf.size());
    }
    importer.FreeScene();
    saveLZ4(path + ".bin", data);
}

void StaticMesh::loadBinary(const std::string & path, Stream & loader) {
    const auto data = loadLZ4(path);
    uint64_t offset = 0;
    uint64_t vertSize;
    read(data,offset, &vertSize);
    PinnedBuffer<Vertex> vertHost(vertSize);
    read(data, offset, vertHost.get(), vertSize);
    vert = MemorySpan<Vertex>(vertSize);
    checkError(cudaMemcpyAsync(vert.begin(),vertHost.get(),vertSize*sizeof(Vertex),
        cudaMemcpyHostToDevice,loader.get()));
    uint64_t faceSize;
    read(data,offset, &faceSize);
    PinnedBuffer<uvec3> indexHost(faceSize);
    read(data, offset, indexHost.get(), faceSize);
    index = MemorySpan<uvec3>(faceSize);
    checkError(cudaMemcpyAsync(index.begin(),indexHost.get(),faceSize*sizeof(uvec3),
        cudaMemcpyHostToDevice,loader.get()));
    loader.sync();
}

void StaticMesh::load(const std::string& path, Stream& loader) {
    if (!std::experimental::filesystem::exists(path + ".bin"))
        convertToBinary(path);
    loadBinary(path + ".bin",loader);
}
