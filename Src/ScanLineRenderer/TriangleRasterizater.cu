#include <ScanLineRenderer/TriangleRasterizater.hpp>

CUDA bool testPoint(mat3 w0, vec2 p, int flag, vec3& w) {
    w.x = w0[0].x*p.x + w0[0].y*p.y+w0[0].z;
    w.y = w0[1].x*p.x + w0[1].y*p.y+w0[1].z;
    w.z = w0[2].x*p.x + w0[2].y*p.y+w0[2].z;
    return (w.x == 0.0f ? (flag & 0b001) : (w.x > 0.0f))
        & (w.y == 0.0f ? (flag & 0b010) : (w.y > 0.0f))
        & (w.z == 0.0f ? (flag & 0b100) : (w.z > 0.0f));
}

CUDA bool calcWeight(mat3 w0, vec2 p, vec3 invz, int flag, vec3& w) {
    bool res = testPoint(w0, p, flag, w);
    w /= dot(invz, w);
    w *= invz;
    return res;
}

