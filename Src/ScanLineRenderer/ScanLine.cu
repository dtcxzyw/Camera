#include <ScanLineRenderer/ScanLine.hpp>

CUDA void toNDC(vec4 & p, uvec2 size) {
    p /= p.w;
    p.x = (0.5f+p.x*0.5f)*size.x, p.y = (0.5f- p.y*0.5f)*size.y;
    p.z += epsilon<float>();
}
