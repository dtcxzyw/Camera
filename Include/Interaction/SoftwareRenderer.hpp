#pragma once
#include <Base/Builtin.hpp>

class ImDrawData;

class SoftwareRenderer final :public Singletion<SoftwareRenderer> {
    friend class Singletion<SoftwareRenderer>;
    SoftwareRenderer() = default;
    std::unique_ptr<BuiltinArray<float>> mTexture;
    std::unique_ptr<BuiltinSampler<float>> mSampler;
    void renderDrawLists(ImDrawData* drawData, CommandBuffer& buffer, 
        BuiltinRenderTarget<RGBA8>& renderTarget);
public:
    void init(Stream& resLoader);
    void uninit();
    void render(CommandBuffer& buffer, BuiltinRenderTarget<RGBA8>& renderTarget);
};
