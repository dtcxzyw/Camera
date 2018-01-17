#include <IO/Model.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <vector>

void StaticMesh::load(const std::string & path,Stream& loader) {
    Assimp::Importer importer;
    auto scene = importer.ReadFile(path, aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType |
        aiProcess_GenSmoothNormals |
        aiProcess_GenUVCoords|
        aiProcess_CalcTangentSpace|
        aiProcess_FixInfacingNormals
    );

    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE)
        throw std::exception("Failed to load the scene.");
    auto mesh=scene->mMeshes[0];
    {
        mVert = allocBuffer<Vertex>(mesh->mNumVertices);
        std::vector<Vertex> vert(mesh->mNumVertices);
        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            vert[i].pos = *reinterpret_cast<vec3*>(mesh->mVertices + i);
            vert[i].normal = *reinterpret_cast<vec3*>(mesh->mNormals + i);
            vert[i].uv = *reinterpret_cast<UV*>(mesh->mTextureCoords[0] + i);
            vert[i].tangent = *reinterpret_cast<vec3*>(mesh->mTangents+i);
        }
        checkError(cudaMemcpyAsync(mVert.begin(),vert.data(),sizeof(Vertex)*vert.size(),
            cudaMemcpyHostToDevice,loader.getID()));
        loader.sync();
    }
    {
        mIndex = allocBuffer<uvec3>(mesh->mNumFaces);
        std::vector<uvec3> index(mesh->mNumFaces);
        for (uint i = 0; i < mesh->mNumFaces; ++i)
            index[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
        checkError(cudaMemcpyAsync(mIndex.begin(), index.data(), sizeof(uvec3)*index.size(),
            cudaMemcpyHostToDevice, loader.getID()));
        loader.sync();
    }
    importer.FreeScene();
}
