#pragma once
#include  <Math/Math.hpp>
#include <Core/Memory.hpp>
#include <Core/Pipeline.hpp>
#include <string>

struct StaticMesh final {
    struct Vertex final {
        ALIGN vec3 pos;
        ALIGN vec3 normal;
        ALIGN vec3 tangent;
        ALIGN UV uv;
    };
    MemorySpan<Vertex> vert;
    MemorySpan<uvec3> index;
    static void convertToBinary(const std::string& path);
    void loadBinary(const std::string& path, Stream& loader);
    void load(const std::string& path, Stream& loader);
};
