#include <ScanLineRenderer/TriangleRasterizater.hpp>

CUDA float edgeFunction(vec4 a, vec4 b, vec4 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

CUDA bool testPoint(mat3 w0, vec2 p, int flag, vec3& w) {
    w.x = w0[0].x + w0[0].y*p.x + w0[0].z*p.y;
    w.y = w0[1].x + w0[1].y*p.x + w0[1].z*p.y;
    w.z = w0[2].x + w0[2].y*p.x + w0[2].z*p.y;
    return (w.x == 0.0f ? (flag & 0b001) : (w.x > 0.0f))
        & (w.y == 0.0f ? (flag & 0b010) : (w.y > 0.0f))
        & (w.z == 0.0f ? (flag & 0b100) : (w.z > 0.0f));
}

CUDA bool calcWeight(mat3 w0, vec2 p, vec3 info, int flag, vec3& w) {
    bool res = testPoint(w0, p, flag, w);
    w *= info / dot(info, w);
    return res;
}

CUDA vec3 calcBase(vec2 a, vec2 b) {
    vec3 res = { 0.0f,b.y - a.y,a.x - b.x };
    res.x = -(a.y * res.z + a.x* res.y);
    return res;
}
