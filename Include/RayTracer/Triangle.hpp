#pragma once
#include <Math/Geometry.hpp>

struct SurfaceInteraction;

class TriangleDesc final {
private:
    DEVICE bool interscet(const Ray& ray, Vector& pa, Vector& pb, Vector& pc,
        float& t, float& e0, float& e1, float& e2, float tMax) const;
public:
    uint32_t id;
    VertexDesc a, b, c;

    DEVICE TriangleDesc(uint32_t id, const VertexDesc& a, const VertexDesc& b,
        const VertexDesc& c);
    DEVICE bool intersect(const Ray& ray, float& tHit, SurfaceInteraction& interaction) const;
    DEVICE bool intersect(const Ray& ray) const;
};

struct TriangleRef final {
    uint32_t id, a, b, c;
    TriangleRef(uint32_t id, uint32_t a, uint32_t b, uint32_t c);
};
