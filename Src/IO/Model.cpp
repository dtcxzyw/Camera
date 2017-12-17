#include <IO/Model.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

bool StaticMesh::load(const std::string & path) {
    Assimp::Importer loader;
    auto scene = loader.ReadFile(path, aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType |
        aiProcess_GenSmoothNormals |
        aiProcess_GenUVCoords|
        aiProcess_CalcTangentSpace|
        aiProcess_FixInfacingNormals
    );

    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE)
        return false;
    auto mesh=scene->mMeshes[0];
    {
        mVert = allocBuffer<Vertex>(mesh->mNumVertices);
        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            mVert[i].pos = *reinterpret_cast<vec3*>(mesh->mVertices + i);
            mVert[i].normal = *reinterpret_cast<vec3*>(mesh->mNormals + i);
            mVert[i].uv = *reinterpret_cast<UV*>(mesh->mTextureCoords[0] + i);
            mVert[i].tangent = *reinterpret_cast<vec3*>(mesh->mTangents+i);
            mVert[i].biTangent = *reinterpret_cast<vec3*>(mesh->mBitangents + i);
        }
    }
    {
        mIndex = allocBuffer<uvec3>(mesh->mNumFaces);
        for (uint i = 0; i < mesh->mNumFaces; ++i)
            mIndex[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
    }
    loader.FreeScene();
    return true;
}
