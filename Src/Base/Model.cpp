#include <Base/Model.hpp>
#include <stb/stb_truetype.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

bool StaticMesh::load(const std::string & path) {
    Assimp::Importer loader;
    auto scene = loader.ReadFile(path, aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType |
        0);
    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE)
        return false;
    auto mesh=scene->mMeshes[0];
    {
        std::vector<Vertex> vert(mesh->mNumVertices);
        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            vert[i].pos = *reinterpret_cast<vec3*>(mesh->mVertices + i);
            vert[i].uv = *reinterpret_cast<vec2*>(mesh->mTextureCoords[0] + i);
        }
        mVert = share(vert);
    }
    {
        std::vector<uvec3> index(mesh->mNumFaces);
        for (uint i = 0; i < mesh->mNumFaces; ++i)
            index[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
        mIndex = share(index);
    }
    auto mat = scene->mMaterials[mesh->mMaterialIndex];
    aiString texp;
    auto tex = mat->GetTexture(aiTextureType_DIFFUSE, 0, &texp);
    mTex = builtinLoadRGBA(texp.C_Str());
    loader.FreeScene();
    return true;
}
