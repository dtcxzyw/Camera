#include <IO/Model.hpp>

#include <Base/CompileBegin.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <Base/CompileEnd.hpp>

void StaticMesh::load(const std::string & path,Stream& loader) {
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
        throw std::runtime_error("Failed to load the scene.");
    const auto mesh=scene->mMeshes[0];
    {
        vert = allocBuffer<Vertex>(mesh->mNumVertices);
        PinnedBuffer<Vertex> temp(mesh->mNumVertices);
        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            temp[i].pos = *reinterpret_cast<vec3*>(mesh->mVertices + i);
            temp[i].normal = *reinterpret_cast<vec3*>(mesh->mNormals + i);
            temp[i].uv = *reinterpret_cast<UV*>(mesh->mTextureCoords[0] + i);
            temp[i].tangent = *reinterpret_cast<vec3*>(mesh->mTangents+i);
        }
        checkError(cudaMemcpyAsync(vert.begin(),temp.get(),sizeof(Vertex)*vert.size(),
            cudaMemcpyHostToDevice,loader.getID()));
        loader.sync();
    }
    {
        index = allocBuffer<uvec3>(mesh->mNumFaces);
        PinnedBuffer<uvec3> temp(mesh->mNumFaces);
        for (uint i = 0; i < mesh->mNumFaces; ++i)
            temp[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
        checkError(cudaMemcpyAsync(index.begin(), temp.get(), sizeof(uvec3)*index.size(),
            cudaMemcpyHostToDevice, loader.getID()));
        loader.sync();
    }
    importer.FreeScene();
}
