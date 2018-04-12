#pragma once
#include <Core/Builtin.hpp>

class SoftwareRenderer final :public Singletion<SoftwareRenderer> {
    friend class Singletion<SoftwareRenderer>;
    SoftwareRenderer() = default;
    std::unique_ptr<BuiltinArray<float>> mTexture;
    std::unique_ptr<BuiltinSampler<float>> mSampler;
public:
    void init(Stream& resLoader);
    void uninit();
    void render(CommandBuffer& buffer, BuiltinRenderTarget<RGBA8>& renderTarget);
};
