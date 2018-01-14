#pragma once
#include <Base/Memory.hpp>
#include <string>

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
    void load(const std::string& path);
};
