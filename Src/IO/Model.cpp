#include <IO/Model.hpp>
#include <Core/IncludeBegin.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <Core/IncludeEnd.hpp>
#include <IO/LZ4Binary.hpp>
#include <filesystem>

template <typename T>
void read(const std::vector<char>& stream, uint64_t& offset, T* ptr, const size_t size = 1) {
    const auto rsiz = sizeof(T) * size;
    memcpy(static_cast<void*>(ptr), stream.data() + offset, rsiz);
    offset += rsiz;
}

template <typename T>
void write(std::vector<char>& stream, const T* ptr, const size_t size = 1) {
    const auto begin = reinterpret_cast<const char*>(ptr);
    stream.insert(stream.end(), begin, begin + sizeof(T) * size);
}

StaticMesh::StaticMesh(const std::string& path) {
    if (!std::experimental::filesystem::exists(path + ".bin"))
        convertToBinary(path);
    loadBinary(path + ".bin");
}

void StaticMesh::convertToBinary(const std::string& path) {
    Assimp::Importer importer;
    const auto scene = importer.ReadFile(path, aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType |
        aiProcess_GenSmoothNormals |
        aiProcess_GenUVCoords |
        aiProcess_CalcTangentSpace |
        aiProcess_FixInfacingNormals |
        aiProcess_ImproveCacheLocality
    );

    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE)
        throw std::runtime_error("Failed to load the mesh.");
    const auto mesh = scene->mMeshes[0];
    std::vector<char> data;
    {
        std::vector<VertexDesc> buf(mesh->mNumVertices);
        for (auto i = 0U; i < mesh->mNumVertices; ++i) {
            buf[i].pos = *reinterpret_cast<Point*>(mesh->mVertices + i);
            buf[i].normal = *reinterpret_cast<Normal*>(mesh->mNormals + i);
            buf[i].tangent = *reinterpret_cast<Normal*>(mesh->mTangents + i);
            buf[i].uv = *reinterpret_cast<UV*>(mesh->mTextureCoords[0] + i);
        }
        const uint64_t vertSize = mesh->mNumVertices;
        write(data, &vertSize);
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

void StaticMesh::loadBinary(const std::string& path) {
    const auto data = loadLZ4(path);
    uint64_t offset = 0;
    uint64_t vertSize;
    read(data, offset, &vertSize);
    vert.resize(vertSize);
    read(data, offset, vert.data(), vertSize);
    uint64_t faceSize;
    read(data, offset, &faceSize);
    index.resize(faceSize);
    read(data, offset, index.data(), faceSize);
}

StaticMeshData::StaticMeshData(const StaticMesh& data, Stream& loader) {
    vert = loader.upload(data.vert);
    index = loader.upload(data.index);
    loader.sync();
}
