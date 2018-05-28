#pragma once
#include <Core/Common.hpp>
#include <memory>
#include <map>
#include <vector>
#include <set>

class Component : Uncopyable {
public:
    virtual size_t getType() const = 0;
    virtual ~Component() = default;
};

class Entity final : Uncopyable {
private:
    std::map<size_t, std::unique_ptr<Component>> mComponents;
public:
    void attachComponent(std::unique_ptr<Component> ptr);

    template <typename T>
    T& getCompontnt() {
        return dynamic_cast<T&>(mComponents[T::getTypeID()]);
    }

    template <typename T>
    bool hasComponent() const {
        return mComponents.count(T::getTypeID());
    }

    bool hasTuple(const std::vector<size_t>& tuple) const;
};

class System : Uncopyable {
private:
    std::vector<size_t> mCompTuple;
    std::set<Entity*> mEntities;

    template <typename First, typename... Comps>
    void addComp() {
        mCompTuple.emplace_back(First::getTypeID());
        addComp<Comps...>();
    }

    static void addComp() noexcept;
public:
    template <typename... Comps>
    System() {
        addComp<Comps...>();
    }

    const std::vector<size_t>& getTuple() const;
    void addEntity(Entity* ptr);
    void eraseEntity(Entity* ptr);
    virtual void update(float delta) =0;
};

using SharedEntity = std::shared_ptr<Entity>;

class Scene final : Uncopyable {
private:
    std::vector<std::unique_ptr<System>> mSystems;
    std::set<SharedEntity> mEntities;
public:
    void addSystem(std::unique_ptr<System> system);
    void addEntity(const SharedEntity& entity);
    void eraseEntity(const SharedEntity& entity);
    void update(float delta);
};
