#pragma once
#include <RayTracer/Interaction.hpp>
#include <Core/DeviceMemory.hpp>

class Primitive {
private:
    CUDA virtual bool intersectImpl(const Ray& ray, float& t, Interaction& interaction) const = 0;
    CUDA virtual bool intersectImpl(const Ray& ray) const = 0;
    CUDA virtual Bounds boundsImpl() const = 0;
public:
    virtual ~Primitive() = default;
    Transform toLocal;
    CUDA bool intersect(const Ray& ray, float& t, Interaction& interaction) const {
        const auto res = intersectImpl(toLocal(ray), t, interaction);
        transform(inverse(toLocal), interaction);
        return res;
    }
    CUDA bool intersect(const Ray& ray) const {
        return intersectImpl(toLocal(ray));
    }
    CUDA Bounds bounds() const {
        return inverse(toLocal)(boundsImpl());
    }
};

class Node final :public Primitive {
private:
    VectorDevice<UniquePtr<Primitive>> mGroup;
    CUDA bool intersectImpl(const Ray& ray) const override {
        for (auto i = 0U; i < mGroup.size(); ++i)
            if (mGroup[i]->intersect(ray))
                return true;
        return false;
    }
    CUDA bool intersectImpl(const Ray& ray, float& t, Interaction& interaction) const override {
        auto res = false;
        for (auto i = 0U; i < mGroup.size(); ++i)
            res |= mGroup[i]->intersect(ray, t, interaction);
        return res;
    }
public:
    CUDA explicit Node(VectorDevice<UniquePtr<Primitive>> group)
        :mGroup(move(group)) {}
};
