#pragma once
#include <Base/Memory.hpp>
#include <string>
#include <Base/Pipeline.hpp>

struct StaticMesh final {
    struct Vertex final {
        ALIGN vec3 pos;
        ALIGN vec3 normal;
        ALIGN vec3 tangent;
        ALIGN UV uv;
    };
    DataViewer<Vertex> mVert;
    DataViewer<uvec3> mIndex;
    void load(const std::string& path,Stream& loader);
};
