#pragma once
#include <Math/Geometry.hpp>

struct Interaction;

class TriangleDesc final {
private:
    CUDA bool interscet(const Ray& ray, Vector& pa, Vector& pb, Vector& pc,
        float& t, float& e0, float& e1, float& e2) const;
public:
    unsigned int id;
    VertexDesc a, b, c;

    CUDA TriangleDesc(unsigned int id, const VertexDesc& a, const VertexDesc& b,
        const VertexDesc& c);
    CUDA bool intersect(const Ray& ray, float& tHit, Interaction& interaction) const;
    CUDA bool intersect(const Ray& ray) const;
};

struct TriangleRef final {
    unsigned int id, a, b, c;
    TriangleRef(unsigned int id, unsigned int a, unsigned int b, unsigned int c);
};
