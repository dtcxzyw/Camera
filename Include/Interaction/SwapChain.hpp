#pragma once
#include <Core/Common.hpp>
#include <vector>
#include <memory>

template <typename Frame>
class SwapChain final : Uncopyable {
private:
    std::vector<std::shared_ptr<Frame>> mImages;
public:
    using SharedFrame = std::shared_ptr<Frame>;

    explicit SwapChain(const size_t size) {
        for (size_t i = 0; i < size; ++i)
            mImages.emplace_back(std::make_shared<Frame>());
    }

    SharedFrame pop() {
        if (mImages.empty())return nullptr;
        auto ptr = std::move(mImages.back());
        mImages.pop_back();
        return ptr;
    }

    void push(SharedFrame&& image) {
        mImages.emplace_back(std::move(image));
    }

    bool empty() const {
        return mImages.empty();
    }
};
