#pragma once
#include <Core/Memory.hpp>
#include <RayTracer/Triangle.hpp>

struct BvhNode {
    Bounds bounds;

    union {
        unsigned int offset;
        unsigned int second;
    };

    unsigned short axis, size;
};

class BvhForTriangleRef final {
private:
    READONLY(BvhNode) mNodes;
    READONLY(TriangleRef) mIndex;
    READONLY(VertexDesc) mVertex;
    DEVICE TriangleDesc makeTriangleDesc(unsigned int id) const;
public:
    BvhForTriangleRef(const MemorySpan<BvhNode>& nodes,
        const MemorySpan<TriangleRef>& index, const MemorySpan<VertexDesc>& vertex);
    DEVICE bool intersect(const Ray& ray) const;
    DEVICE bool intersect(const Ray& ray, float& t, Interaction& interaction) const;
};

class StaticMesh;
class Stream;

class BvhForTriangle final {
private:
    MemorySpan<BvhNode> mNodes;
    MemorySpan<TriangleRef> mIndex;
    MemorySpan<VertexDesc> mVertex;
public:
    BvhForTriangle(const StaticMesh& mesh, size_t maxPrim, Stream& stream);
    BvhForTriangleRef getRef() const;
};
