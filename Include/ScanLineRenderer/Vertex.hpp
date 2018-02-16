#pragma once
#include <Base/Common.hpp>
#include  <Base/Math.hpp>
#include <Base/Memory.hpp>
#include <Base/DispatchSystem.hpp>

class UniqueIndex final {
private:
    unsigned int mSize;
public:
    explicit UniqueIndex(const unsigned int size) :mSize(size) {}
    BOTH auto size() const {
        return mSize;
    }
    CUDAINLINE uvec3 operator[](const unsigned int off) const {
        const auto base = off * 3;
        return { base,base + 1,base + 2 };
    }
};

class SharedIndex final {
private:
    READONLY(uvec3) mPtr;
    unsigned int mSize;
public:
    SharedIndex(READONLY(uvec3) idx, const unsigned int size) :mPtr(idx), mSize(size) {}
    explicit SharedIndex(const DataViewer<uvec3>& ibo) :mPtr(ibo.begin()), mSize(ibo.size()) {}
    BOTH auto size() const {
        return mSize;
    }
    CUDAINLINE auto operator[](const unsigned int off) const {
        return mPtr[off];
    }
};

template<typename Vert, typename Out, typename Uniform>
using VSF = void(*)(Vert in, const Uniform& uniform, vec3& pos, Out& out);

template<typename Out>
struct VertexInfo {
    vec3 pos;
    /*
    NDC.x=pos.x/pos.z
    NDC.y=pos.y/pos.z
    z=pos.z
    */
    Out out;
};

template<typename Out>
CUDAINLINE VertexInfo<Out> lerpZ(VertexInfo<Out> a, VertexInfo<Out> b, float z) {
    auto u = (z - b.pos.z) / (a.pos.z - b.pos.z), v = 1.0f - u;
    VertexInfo<Out> res;
    res.pos = { a.pos.x*u + b.pos.x*v,a.pos.y*u + b.pos.y*v,z };
    res.out = a.out*u + b.out*v;
    return res;
}

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
GLOBAL void runVS(const unsigned int size, READONLY(Vert) in,
    READONLY(Uniform) u, VertexInfo<Out>* res) {
    const auto id = getID();
    if (id >= size)return;
    auto& vert = res[id];
    vs(in[id], *u, vert.pos, vert.out);
}

template<typename Vert, typename Out, typename Uniform, VSF<Vert, Out, Uniform> vs>
auto calcVertex(CommandBuffer& buffer, const DataPtr<Vert>& vert, const DataPtr<Uniform>& uniform) {
    auto vertex = buffer.allocBuffer<VertexInfo<Out>>(vert.size());
    buffer.runKernelLinear(runVS<Vert, Out, Uniform, vs>, vert.size(),
        vert.get(), uniform.get(), vertex);
    return vertex;
}

CUDAINLINE vec3 toRaster(const vec3 p, const vec2 hsiz) {
    const auto invz = 1.0f / p.z;
    return { (1.0f + p.x*invz)*hsiz.x,(1.0f - p.y*invz)*hsiz.y,invz };
}
