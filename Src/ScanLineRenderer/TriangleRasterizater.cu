#include <ScanLineRenderer/TriangleRasterizater.hpp>

CUDA float edgeFunction(vec4 a, vec4 b, vec4 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

CUDA bool testPoint(vec4 a, vec4 b, vec4 c, vec4 p,int flag, vec3& w) {
    w.x = edgeFunction(b, c, p), w.y = edgeFunction(c, a, p), w.z = edgeFunction(a, b, p);
    return (w.x == 0.0f ? (flag&0b001) : (w.x > 0.0f))
        & (w.y == 0.0f ? (flag&0b010) : (w.y > 0.0f))
        & (w.z == 0.0f ? (flag&0b100) : (w.z > 0.0f));
}

CUDA bool calcWeight(vec4 a, vec4 b, vec4 c, vec4 p, vec3 info,int flag, vec3& w) {
    bool res = testPoint(a, b, c, p,flag, w);
    w *= info / dot(info,w);
    return res;
}
