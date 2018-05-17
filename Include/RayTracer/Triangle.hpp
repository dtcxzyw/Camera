#pragma once
#include <Math/Geometry.hpp>

struct Interaction;

class TriangleDesc final {
private:
    DEVICE bool interscet(const Ray& ray, Vector& pa, Vector& pb, Vector& pc,
        float& t, float& e0, float& e1, float& e2) const;
public:
    unsigned int id;
    VertexDesc a, b, c;

    DEVICE TriangleDesc(unsigned int id, const VertexDesc& a, const VertexDesc& b,
        const VertexDesc& c);
    DEVICE bool intersect(const Ray& ray, float& tHit, Interaction& interaction) const;
    DEVICE bool intersect(const Ray& ray) const;
};

struct TriangleRef final {
    unsigned int id, a, b, c;
    TriangleRef(unsigned int id, unsigned int a, unsigned int b, unsigned int c);
};
