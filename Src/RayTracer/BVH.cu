#include <RayTracer/BVH.hpp>
#include <algorithm>
#include <IO/Model.hpp>

DEVICE TriangleDesc BvhForTriangleRef::makeTriangleDesc(const unsigned int id) const {
    const auto ref = mIndex[id];
    return {ref.id, mVertex[ref.a], mVertex[ref.b], mVertex[ref.c]};
}

BvhForTriangleRef::BvhForTriangleRef(const MemorySpan<BvhNode>& nodes, 
    const MemorySpan<TriangleRef>& index,const MemorySpan<VertexDesc>& vertex)
    :mNodes(nodes.begin()), mIndex(index.begin()), mVertex(vertex.begin()) {}

DEVICE bool BvhForTriangleRef::intersect(const Ray& ray) const {
    unsigned int top = 0, current = 0;
    unsigned int stack[64];
    const auto invDir = 1.0f / ray.dir;
    const glm::bvec3 neg = {invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f};
    while (true) {
        const auto& node = mNodes[current];
        if (node.bounds.intersect(ray, ray.tMax, invDir, neg)) {
            if (node.size) {
                for (auto i = 0U; i < node.size; ++i)
                    if (makeTriangleDesc(node.offset + i).intersect(ray))
                        return true;
                if (top == 0)break;
                current = stack[--top];
            }
            else if (neg[node.axis]) {
                stack[top++] = current + 1;
                current = node.second;
            }
            else {
                stack[top++] = node.second + 1;
                ++current;
            }
        }
        else {
            if (top == 0)break;
            current = stack[--top];
        }
    }
    return false;
}

DEVICE bool BvhForTriangleRef::intersect(const Ray& ray, float& t, Interaction& interaction) const {
    unsigned int top = 0, current = 0;
    unsigned int stack[64];
    const auto invDir = 1.0f / ray.dir;
    const glm::bvec3 neg = {invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f};
    auto res = false;
    while (true) {
        const auto& node = mNodes[current];
        if (node.bounds.intersect(ray,t, invDir, neg)) {
            if (node.size) {
                for (auto i = 0U; i < node.size; ++i)
                    res |= makeTriangleDesc(node.offset + i).intersect(ray, t, interaction);
                if (top == 0)break;
                current = stack[--top];
            }
            else if (neg[node.axis]) {
                stack[top++] = current + 1;
                current = node.second;
            }
            else {
                stack[top++] = node.second + 1;
                ++current;
            }
        }
        else {
            if (top == 0)break;
            current = stack[--top];
        }
    }
    return res;
}

struct BuildNode final {
    Bounds bounds;
    BuildNode* child[2];
    unsigned int axis, offset, size;

    void setLeaf(const unsigned int off, const unsigned int siz, const Bounds& b) {
        bounds = b;
        child[0] = child[1] = nullptr;
        offset = off;
        size = siz;
    }

    void setInterior(const unsigned int axisId, BuildNode* cl, BuildNode* cr) {
        axis = axisId;
        size = 0;
        bounds = cl->bounds | cr->bounds;
        child[0] = cl;
        child[1] = cr;
    }
};

struct PrimitiveInfo final {
    unsigned int id;
    Bounds bounds;
    Point centroid;

    PrimitiveInfo(const unsigned int id, const Bounds& bounds) : id(id), bounds(bounds),
        centroid(mix(bounds[0], bounds[1], 0.5f)) {}
};

BuildNode* buildTriangleRecursive(std::vector<BuildNode>& nodePool, const std::vector<uvec3>& index,
    std::vector<PrimitiveInfo>& info, const size_t begin, const size_t end,
    std::vector<TriangleRef>& ordered, const size_t maxPrim) {
    nodePool.emplace_back();
    auto node = &nodePool.back();
    Bounds bounds;
    for (auto i = begin; i < end; ++i)
        bounds |= info[i].bounds;
    const auto size = end - begin;
    const auto initLeaf = [&]() {
        node->setLeaf(ordered.size(), size, bounds);
        for (auto i = begin; i < end; ++i) {
            const auto idx = index[info[i].id];
            ordered.emplace_back(info[i].id, idx.x, idx.y, idx.z);
        }
    };
    if (size == 1U) initLeaf();
    else {
        Bounds centroidBounds;
        for (auto i = begin; i < end; ++i)
            centroidBounds |= Bounds(info[i].bounds);
        const auto axis = maxDim(centroidBounds[1] - centroidBounds[0]);
        if (centroidBounds[0][axis] == centroidBounds[1][axis])initLeaf();
        else {
            auto mid = (begin + end) / 2;
            //SAH
            if (size <= 4) {
                std::nth_element(info.begin() + begin, info.begin() + mid, info.begin() + end,
                    [axis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                        return a.centroid[axis] < b.centroid[axis];
                    });
            }
            else {
                constexpr auto bucketSize = 16U;
                struct Bucket final {
                    unsigned int count;
                    Bounds bounds;
                } buckets[bucketSize];
                const auto offset = centroidBounds[0];
                const auto inv = 1.0f / (centroidBounds[1] - centroidBounds[0]);
                for (auto i = begin; i < end; ++i) {
                    unsigned int p = bucketSize * ((info[i].centroid - offset) * inv)[axis];
                    p = clamp(p, 0U, bucketSize - 1U);
                    ++buckets[p].count;
                    buckets[p].bounds |= info[i].bounds;
                }
                auto minCost = std::numeric_limits<float>::max();
                unsigned int splitPos;
                for (auto i = 0U; i < bucketSize - 1U; ++i) {
                    Bounds partBounds[2];
                    size_t partCount[2] = {};
                    for (auto j = 0U; j < bucketSize; ++j) {
                        const auto part = j > i;
                        partBounds[part] |= buckets[j].bounds;
                        partCount[part] += buckets[j].count;
                    }
                    const auto cost = 0.125f + (partCount[0] * partBounds[0].area() +
                        partCount[1] * partBounds[1].area()) / bounds.area();
                    if (minCost > cost) {
                        splitPos = i;
                        minCost = cost;
                    }
                }
                const float leafCost = size;
                if (size > maxPrim || minCost < leafCost) {
                    mid = std::partition(info.begin() + begin, info.begin() + end,
                        [=](const PrimitiveInfo& prim) {
                            unsigned int p = bucketSize * ((prim.centroid - offset) * inv)[axis];
                            p = clamp(p, 0U, bucketSize - 1U);
                            return p <= splitPos;
                        }) - info.begin();
                }
                else {
                    initLeaf();
                    return node;
                }
            }

            node->setInterior(axis,
                buildTriangleRecursive(nodePool, index, info, begin, mid, ordered, maxPrim),
                buildTriangleRecursive(nodePool, index, info, mid, end, ordered, maxPrim));
        }
    }
    return node;
}

size_t flattenTree(const BuildNode* node, size_t& offset,
    PinnedBuffer<BvhNode>& buf) {
    auto& self = buf[offset];
    self.bounds = node->bounds;
    const auto off = offset++;
    if (node->size) {
        self.offset = node->offset;
        self.size = node->size;
    }
    else {
        self.axis = node->axis;
        self.size = 0;
        flattenTree(node->child[0], offset, buf);
        self.second = flattenTree(node->child[1], offset, buf);
    }
    return off;
}

BvhForTriangle::BvhForTriangle(const StaticMesh& mesh, const size_t maxPrim,Stream& stream) {
    auto&& vertex = mesh.vert;
    auto&& index = mesh.index;
    mVertex = MemorySpan<VertexDesc>(vertex.size());
    checkError(cudaMemcpyAsync(mVertex.begin(), vertex.data(), sizeof(VertexDesc)*vertex.size(),
        cudaMemcpyHostToDevice, stream.get()));

    std::vector<PrimitiveInfo> primitive;
    primitive.reserve(index.size());
    for (const auto idx : index) {
        const auto pa = vertex[idx.x].pos, pb = vertex[idx.y].pos, pc = vertex[idx.z].pos;
        Bounds bounds{min(pa, min(pb, pc)), max(pa, max(pb, pc))};
        primitive.emplace_back(static_cast<unsigned int>(primitive.size()), bounds);
    }
    std::vector<TriangleRef> ordered;
    ordered.reserve(index.size());
    std::vector<BuildNode> nodePool;
    nodePool.reserve(index.size() * 2);
    const auto root = buildTriangleRecursive(nodePool, index, primitive, 0, primitive.size(), ordered,
        maxPrim);

    mIndex = MemorySpan<TriangleRef>(ordered.size());
    checkError(cudaMemcpyAsync(mIndex.begin(), ordered.data(), sizeof(TriangleRef) * mIndex.size(),
        cudaMemcpyHostToDevice, stream.get()));

    PinnedBuffer<BvhNode> nodes(nodePool.size());
    size_t offset = 0U;
    flattenTree(root, offset, nodes);
    mNodes = MemorySpan<BvhNode>(nodes.size());
    checkError(cudaMemcpyAsync(mNodes.begin(), nodes.get(), sizeof(BvhNode) * mNodes.size(),
        cudaMemcpyHostToDevice, stream.get()));
    stream.sync();
}

BvhForTriangleRef BvhForTriangle::getRef() const {
    return { mNodes,mIndex,mVertex };
}
