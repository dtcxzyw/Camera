#pragma once
#include <Base/Common.hpp>

struct StaticMesh final {
    struct Vertex {
        ALIGN vec3 pos;
        ALIGN vec3 normal;
        ALIGN vec3 tangent;
        ALIGN vec3 biTangent;
        ALIGN UV uv;
    };
    DataViewer<Vertex> mVert;
    DataViewer<uvec3> mIndex;
    bool load(const std::string& path);
};
