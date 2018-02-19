#include <Scene/ECS.hpp>

void Entity::attachComponent(std::unique_ptr<Component> ptr) {
    mComponents[ptr->getType()] = std::move(ptr);
}

bool Entity::hasTuple(const std::vector<size_t>& tuple) const {
    for (auto&& id : tuple)
        if (mComponents.count(id) == 0)
            return false;
    return true;
}

void System::addComp() noexcept {}

const std::vector<size_t>& System::getTuple() const {
    return mCompTuple;
}

void System::addEntity(Entity* ptr) {
    mEntities.emplace(ptr);
}

void System::eraseEntity(Entity* ptr) {
    mEntities.erase(ptr);
}

void Scene::addSystem(std::unique_ptr<System> system) {
    mSystems.emplace_back(std::move(system));
}

void Scene::addEntity(const SharedEntity& entity) {
    mEntities.insert(entity);
    for (auto&& system : mSystems)
        if (entity->hasTuple(system->getTuple()))
            system->addEntity(entity.get());
}

void Scene::eraseEntity(const SharedEntity& entity) {
    mEntities.erase(entity);
    for (auto&& system : mSystems)
        system->eraseEntity(entity.get());
}

void Scene::update(const float delta) {
    for (auto&& system : mSystems)
        system->update(delta);
}
