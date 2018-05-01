#pragma once
#include <RayTracer/Primitive.hpp>
#include <Core/Memory.hpp>
#include <RayTracer/Triangle.hpp>
#include <Core/DeviceMemory.hpp>
#include <IO/Model.hpp>

struct BvhNode {
    Bounds bounds;
    union {
        unsigned int offset;
        unsigned int second;
    };
    unsigned short axis, size;
};

struct BvhForTriangleInfo final {
    BvhNode* nodes;
    TriangleRef* index;
    VertexDesc* vertex;
    Bounds bounds;
};

class BvhForTriangleRef final :public Primitive {
private:
    READONLY(BvhNode) mNodes;
    READONLY(TriangleRef) mIndex;
    READONLY(VertexDesc) mVertex;
    Bounds mBounds;
    CUDA TriangleDesc makeTriangleDesc(unsigned int id) const;
    CUDA bool intersectImpl(const Ray& ray) const override;
    CUDA bool intersectImpl(const Ray& ray, float& t, Interaction& interaction) const override;
    CUDA Bounds boundsImpl() const override;
public:
    CUDA explicit BvhForTriangleRef(const BvhForTriangleInfo& info);
};

class BvhForTriangle final {
private:
    MemorySpan<BvhNode> mNodes;
    MemorySpan<TriangleRef> mIndex;
    MemorySpan<VertexDesc> mVertex;
    Bounds mBounds;
public:
    BvhForTriangle(const StaticMesh& mesh, size_t maxPrim,Stream& stream);
    BvhForTriangleInfo getBuildInfo() const;
};
