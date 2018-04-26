#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <Core/Pipeline.hpp>
#include <string>

struct StaticMesh final {
    struct Vertex final {
        ALIGN Point pos;
        ALIGN Vector normal;
        ALIGN Vector tangent;
        ALIGN UV uv;
    };
    MemorySpan<Vertex> vert;
    MemorySpan<uvec3> index;
    static void convertToBinary(const std::string& path);
    void loadBinary(const std::string& path, Stream& loader);
    void load(const std::string& path, Stream& loader);
};
