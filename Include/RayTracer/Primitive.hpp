#pragma once
#include <RayTracer/Interaction.hpp>

class Primitive {
private:
    CUDA virtual bool intersectImpl(const Ray& ray, float& t, Interaction& interaction) const = 0;
    CUDA virtual bool intersectImpl(const Ray& ray) const = 0;
public:
    Transform toLocal;
    CUDA bool intersect(const Ray& ray, float& t, Interaction& interaction) const {
        return intersectImpl(toLocal(ray), t, interaction);
    }
    CUDA bool intersect(const Ray& ray) const {
        return intersectImpl(toLocal(ray));
    }
};

class PrimitiveGroup final :public Primitive {
private:
    Primitive** mGroup;
    unsigned int mSize;
    CUDA bool intersectImpl(const Ray& ray) const override {
        for (auto i = 0U; i < mSize; ++i)
            if (mGroup[i]->intersectImpl(ray))
                return true;
        return false;
    }
    CUDA bool intersectImpl(const Ray& ray, float& t, Interaction& interaction) const override {
        auto res = false;
        for (auto i = 0U; i < mSize; ++i)
            res |= mGroup[i]->intersectImpl(ray, t, interaction);
        return res;
    }
public:
    CUDA PrimitiveGroup(Primitive** group, const unsigned int size)
        :mGroup(group), mSize(size) {}
};
