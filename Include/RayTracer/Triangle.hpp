#pragma once
#include <RayTracer/Primitive.hpp>

struct TriangleDesc final {
    unsigned int id;
    VertexDesc a, b, c;

    CUDA TriangleDesc(const unsigned int id, const VertexDesc& a, const VertexDesc& b,
        const VertexDesc& c) : id(id), a(a), b(b), c(c) {}

    CUDA bool interscet(const Ray& ray, Vector& pa, Vector& pb, Vector& pc,
        float& t, float& e0, float& e1, float& e2) const {
        pa = a.pos - ray.origin;
        pb = b.pos - ray.origin;
        pc = c.pos - ray.origin;
        const auto z = maxDim(ray.dir);
        const auto x = (z + 1) % 3, y = (x + 1) % 3;
        const auto dir = permute(ray.dir, x, y, z);
        pa = permute(pa, x, y, z);
        pb = permute(pb, x, y, z);
        pc = permute(pc, x, y, z);
        const auto sz = 1.0f / dir.z;
        const auto sx = dir.x * sz;
        const auto sy = dir.y * sz;
        pa.x -= sx * pa.z;
        pa.y -= sy * pa.z;
        pb.x -= sx * pb.z;
        pb.y -= sy * pb.z;
        pc.x -= sx * pc.z;
        pc.y -= sy * pc.z;
        e0 = pb.x * pc.y - pc.x * pb.y;
        e1 = pc.x * pa.y - pa.x * pc.y;
        e2 = pa.x * pb.y - pb.x * pa.y;
        if (min3(e0, e1, e2) * max3(e0, e1, e2) < 0.0f)return false;
        const auto det = e0 + e1 + e2;
        if (det == 0.0f)return false;
        const auto invDet = 1.0f / det;
        t = (e0 * pa.z + e1 * pb.z + e2 * pc.z) * sz * invDet;
        if (t < ray.tMax)return false;
        e0 *= invDet;
        e1 *= invDet;
        e2 *= invDet;
        //TODO : reduce error
        return true;
    }

    CUDA bool intersect(const Ray& ray, float& tHit, Interaction& interaction) const {
        Vector pa, pb, pc;
        float t, e0, e1, e2;
        if (interscet(ray, pa, pb, pc, t, e0, e1, e2)) {
            interaction.pos = a.pos * e0 + b.pos * e1 + c.pos * e2;
            interaction.dir = Normal(-ray.dir);
            const auto dpac = a.pos - c.pos, dpbc = b.pos - c.pos;
            interaction.uv = a.uv * e0 + b.uv * e1 + c.uv * e2;
            const auto duvac = a.uv - c.uv, duvbc = b.uv - c.uv;
            const auto invDet = 1.0f / cross(dpac, dpbc);
            const Normal normal{ a.normal*e0 + b.normal*e1 + c.normal*e2 };
            Normal tangent{ a.tangent*e0 + b.tangent*e1 + c.tangent*e2 };
            const auto biTangent = cross(tangent, normal);
            tangent = cross(biTangent, normal);
            interaction.normal = normal;
            interaction.dpdu = tangent;
            interaction.dpdv = biTangent;
            const auto dnac = a.normal - c.normal;
            const auto dnbc = b.normal - c.normal;
            interaction.dndu = (duvbc.y * dnac - duvac.y * dnbc) * invDet;
            interaction.dndv = (duvac.x * dnbc - duvbc.x * dnac) * invDet;
            tHit = t;
        }
        return false;
    }

    CUDA bool intersect(const Ray& ray) const {
        Vector pa, pb, pc;
        float t, e0, e1, e2;
        return interscet(ray, pa, pb, pc, t, e0, e1, e2);
    }

    CUDA Bounds bounds() const {
        return Bounds(a.pos) | Bounds(b.pos) | Bounds(c.pos);
    }
};

struct TriangleRef final {
    unsigned int id, a, b, c;
    TriangleRef(const unsigned int id, const unsigned int a, const unsigned int b, 
        const unsigned int c) :id(id), a(a), b(b), c(c) {}
};
