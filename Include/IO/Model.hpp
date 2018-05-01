#pragma once
#include <Math/Geometry.hpp>
#include <Core/Memory.hpp>
#include <Core/Pipeline.hpp>
#include <string>
#include <vector>

class StaticMesh final {
public:
    std::vector<VertexDesc> vert;
    std::vector<uvec3> index;
    explicit StaticMesh(const std::string& path);
private:
    static void convertToBinary(const std::string& path);
    void loadBinary(const std::string& path);
};

struct StaticMeshData final {
    MemorySpan<VertexDesc> vert;
    MemorySpan<uvec3> index;
    StaticMeshData(const StaticMesh& data, Stream& loader);
};

