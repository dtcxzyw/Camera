#pragma once
#include <Core/Common.hpp>
#include <Math/Geometry.hpp>

class Primitive {
public:
    mat4 toLocal;
    CUDA virtual Bounds3 bounds() const = 0;
    CUDA virtual bool intersect(const Ray& ray) const = 0;
    CUDA virtual ~Primitive() = default;
};

template<typename T>
class FixedPrimitiveGroup final :public Primitive {
private:
    READONLY(T) mGroup;
    unsigned int mSize;
    Bounds3 mBounds;
public:
    CUDA FixedPrimitiveGroup(READONLY(T) group, const unsigned int size)
        :mGroup(group), mSize(size) {
        mBounds = mGroup[0]->bounds();
        for (auto i = 1U; i < mSize; ++i)
            mBounds |= mGroup[i]->bounds();
    }
    CUDA Bounds3 bounds() const override {
        return mBounds;
    }
    CUDA bool intersect(const Ray& ray) const override {
        for (auto i = 0U; i < mSize; ++i)
            if (mGroup[i]->intersect(ray))
                return true;
        return false;
    }
};

class PrimitiveGroup final :public Primitive {
private:
    Primitive** mGroup;
    unsigned int mSize;
    Bounds3 mBounds;
public:
    CUDA PrimitiveGroup(Primitive** group, const unsigned int size)
        :mGroup(group), mSize(size) {
        mBounds = mGroup[0]->bounds();
        for (auto i = 1U; i < mSize; ++i)
            mBounds |= mGroup[i]->bounds();
    }
    CUDA Bounds3 bounds() const override {
        return mBounds;
    }
    CUDA bool intersect(const Ray& ray) const override {
        for (auto i = 0U; i < mSize; ++i)
            if (mGroup[i]->intersect(ray))
                return true;
        return false;
    }
};
