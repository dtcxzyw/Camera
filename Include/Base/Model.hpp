#pragma once
#include "Common.hpp"
#include "Builtin.hpp"

struct StaticMesh final {
    struct Vertex {
        ALIGN vec3 pos;
        ALIGN vec2 uv;
    };
    DataViewer<Vertex> mVert;
    DataViewer<uvec3> mIndex;
    std::shared_ptr<BuiltinSampler<RGBA>> mTex;
    bool load(const std::string& path);
};
