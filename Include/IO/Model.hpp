#pragma once
#include <Base/Common.hpp>

struct StaticMesh final {
    struct Vertex {
        ALIGN vec3 pos;
    };
    DataViewer<Vertex> mVert;
    DataViewer<uvec3> mIndex;
    bool load(const std::string& path);
};
