#pragma once
#include  <Base/Math.hpp>
#include <Base/Memory.hpp>
#include <Base/Pipeline.hpp>
#include <string>

struct StaticMesh final {
    struct Vertex final {
        ALIGN vec3 pos;
        ALIGN vec3 normal;
        ALIGN vec3 tangent;
        ALIGN UV uv;
    };
    DataViewer<Vertex> vert;
    DataViewer<uvec3> index;
    void load(const std::string& path,Stream& loader);
};
