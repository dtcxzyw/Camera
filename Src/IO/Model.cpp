#include <IO/Model.hpp>
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
        for (uint i = 0; i < mesh->mNumVertices; ++i)
            vert[i].pos = *reinterpret_cast<vec3*>(mesh->mVertices + i);
        mVert = share(vert);
    }
    {
        std::vector<uvec3> index(mesh->mNumFaces);
        for (uint i = 0; i < mesh->mNumFaces; ++i)
            index[i] = *reinterpret_cast<uvec3*>(mesh->mFaces[i].mIndices);
        mIndex = share(index);
    }
    loader.FreeScene();
    return true;
}
