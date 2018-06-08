#include <RayTracer/Triangle.hpp>
#include <Math/Interaction.hpp>

DEVICE bool TriangleDesc::interscet(const Ray& ray, Vector& pa, Vector& pb, Vector& pc,
    float& t, float& e0, float& e1, float& e2, const float tMax) const {
    pa = a.pos - ray.origin;
    pb = b.pos - ray.origin;
    pc = c.pos - ray.origin;
    const auto z = maxDim(ray.dir);
    constexpr int nextDim[3] = {1, 2, 0};
    const auto x = nextDim[z], y = nextDim[x];
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
    if (t >= tMax)return false;

    const auto maxXt = absMax3(pa.x, pb.x, pc.x);
    const auto deltaX = gamma(5) * maxXt;
    const auto maxYt = absMax3(pa.y, pb.y, pc.y);
    const auto deltaY = gamma(5) * maxYt;
    const auto maxZt = absMax3(pa.z, pb.z, pc.z)*fabs(sz);
    const auto deltaZ = gamma(3) * maxZt;
    const auto deltaE = 2.0f * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);
    const auto maxE = absMax3(e0, e1, e2);
    const auto deltaT = 3.0f * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * fabs(invDet);
    if (t <= deltaT) return false;

    e0 *= invDet;
    e1 *= invDet;
    e2 *= invDet;
    return true;
}

DEVICE TriangleDesc::TriangleDesc(const unsigned id, const VertexDesc& a, const VertexDesc& b,
    const VertexDesc& c): id(id), a(a), b(b), c(c) {}

DEVICE void defaultCoordinateSystem(const Vector& n,Vector& t,Vector& b) {
    if (fabs(n.x) > fabs(n.y))
        t = Vector(-n.z, 0.0f, n.x) / std::sqrt(n.x * n.x + n.z * n.z);
    else
        t = Vector(0.0f, n.z, -n.y) / std::sqrt(n.y * n.y + n.z * n.z);
    b = cross(n, t);
}

DEVICE bool TriangleDesc::intersect(const Ray& ray, float& tHit, Interaction& interaction) const {
    Vector pa, pb, pc;
    float t, e0, e1, e2;
    if (interscet(ray, pa, pb, pc, t, e0, e1, e2, tHit)) {
        interaction.pos = a.pos * e0 + b.pos * e1 + c.pos * e2;
        interaction.pError = gamma(7)*Vector(
            (fabs(e0 * pa.x) + fabs(e1 * pb.x) + fabs(e2 * pc.x)),
            (fabs(e0 * pa.y) + fabs(e1 * pb.y) + fabs(e2 * pc.y)),
            (fabs(e0 * pa.z) + fabs(e1 * pb.z) + fabs(e2 * pc.z))
        );
        interaction.wo = Normal(-ray.dir);
        const auto dpac = a.pos - c.pos, dpbc = b.pos - c.pos;
        interaction.uv = a.uv * e0 + b.uv * e1 + c.uv * e2;
        interaction.id = id;
        const auto duvac = a.uv - c.uv, duvbc = b.uv - c.uv;
        const auto det = duvac.x*duvbc.y - duvbc.x*duvac.y;
        auto&& local = interaction.localGeometry;
        auto&& shading = interaction.shadingGeometry;
        if (fabs(det) > 1e-8f) {
            const auto invDet = 1.0f / det;
            {
                local.dpdu = (duvbc.y * dpac - duvac.y * dpbc) * invDet;
                local.dpdv = (duvac.x * dpbc - duvbc.x * dpac) * invDet;
                local.dndu = local.dndv = {};
            }
            {
                const auto dnac = a.normal - c.normal;
                const auto dnbc = b.normal - c.normal;
                shading.dndu = (duvbc.y * dnac - duvac.y * dnbc) * invDet;
                shading.dndv = (duvac.x * dnbc - duvbc.x * dnac) * invDet;
            }
        }
        else {
            const auto n = cross(pc - pa, pb - pa);
            if (length2(n) == 0.0f)return false;
            defaultCoordinateSystem(glm::normalize(n), local.dpdu, local.dpdv);
            local.dndu = local.dndv = {};
            const auto dn = cross(Vector(c.normal - a.normal), Vector(b.normal - a.normal));
            if (glm::length2(dn) == 0.0f)shading.dndu = shading.dndv = {};
            else defaultCoordinateSystem(dn, shading.dndu, shading.dndv);
        }
        {
            local.normal = faceForward(Normal(cross(local.dpdu, local.dpdv)), shading.normal);
            shading.normal = Normal{ a.normal * e0 + b.normal * e1 + c.normal * e2 };
            const Normal tangent{ a.tangent * e0 + b.tangent * e1 + c.normal * e2 };
            const auto biTangent = cross(tangent, shading.normal);
            shading.dpdu = Vector{ cross(biTangent, shading.normal) };
            shading.dpdv = Vector{ biTangent };
        }

        tHit = t;
        return true;
    }
    return false;
}

DEVICE bool TriangleDesc::intersect(const Ray& ray) const {
    Vector pa, pb, pc;
    float t, e0, e1, e2;
    return interscet(ray, pa, pb, pc, t, e0, e1, e2, ray.tMax);
}

TriangleRef::TriangleRef(const unsigned id, const unsigned a, const unsigned b, const unsigned c)
    : id(id), a(a), b(b), c(c) {}
