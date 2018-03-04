#include <Interaction/SoftwareRenderer.hpp>
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <Base/CompileEnd.hpp>
#include <ScanLineRenderer/TriangleRasterizer.hpp>
#include <Base/DataSet.hpp>
#include <ScanLineRenderer/IndexDescriptor.hpp>

enum class VertOutAttr {
    TexCoord,Color
};

using VertOut = Args<VAR(VertOutAttr::TexCoord, vec2), VAR(VertOutAttr::Color, RGBA)>;

constexpr auto int2Float = 1.0f / 255.0f;

vec4 toRGBA(const unsigned int col) {
    return vec4{ col & 0xff,(col >> 8) & 0xff,(col >> 16) & 0xff,col >> 24 }*int2Float;
}

struct VertInfo final{
    ALIGN vec2 pos;
    ALIGN vec2 uv;
    ALIGN vec4 col;
};

CUDAINLINE void vertShader(VertInfo in, const Empty&, vec3& cpos, 
    VertOut& out) {
    cpos = { in.pos.x,in.pos.y,1.0f };
    out.get<VertOutAttr::TexCoord>() = { in.uv.x,in.uv.y };
    out.get<VertOutAttr::Color>() = in.col;
}

CUDAINLINE bool clipShader(unsigned int, vec3&, vec3&, vec3&, const BuiltinSamplerGPU<float>&) {
    return true;
}

CUDAINLINE void fragShader(unsigned int, ivec2 uv, float, const VertOut& in, const VertOut&, const VertOut&,
    const BuiltinSamplerGPU<float>& texture, BuiltinRenderTargetGPU<RGBA8>& fbo) {
    const auto texAlpha = texture.get(in.get<VertOutAttr::TexCoord>());
    auto src = in.get<VertOutAttr::Color>();
    src.a *= texAlpha;
    const auto dst = vec4(fbo.get(uv))*int2Float;
    const auto alpha = src.a, invAlpha = 1.0f - alpha;
    const auto col = src*alpha + dst * invAlpha;
    const auto res = clamp(col, 0.0f, 1.0f);
    //const auto res = clamp(vec4{ vec3{src},1.0f }, 0.0f, 1.0f);
    fbo.set(uv, RGBA8{res*253.0f});
}

void SoftwareRenderer::renderDrawLists(ImDrawData* drawData,CommandBuffer& buffer,
    BuiltinRenderTarget<RGBA8>& renderTarget) {
    auto&& io = ImGui::GetIO();
    const int fbw = io.DisplaySize.x * io.DisplayFramebufferScale.x;
    const int fbh = io.DisplaySize.y * io.DisplayFramebufferScale.y;
    if (fbw == 0 || fbh == 0)return;
    drawData->ScaleClipRects(io.DisplayFramebufferScale);

    auto uni=buffer.allocConstant<BuiltinSamplerGPU<float>>();
    buffer.memcpy(uni, [this](auto&& call) {
        const auto data = mSampler->toSampler();
        call(&data);
    });

    auto frameBuffer = buffer.allocConstant<BuiltinRenderTargetGPU<RGBA8>>();
    buffer.memcpy(frameBuffer,[rt=renderTarget.toTarget()](auto&& call) {
        call(&rt);
    });

    auto vbo = buffer.allocBuffer<VertInfo>(drawData->TotalVtxCount);
    {
        const auto mul = 2.0f / static_cast<vec2>(renderTarget.size());
        std::vector<VertInfo> tmp(vbo.size());
        auto idx = 0U;
        for(auto i = 0; i < drawData->CmdListsCount; ++i){
            const auto cmdList = drawData->CmdLists[i];
            for (const auto& vert : cmdList->VtxBuffer) {
                auto&& res = tmp[idx++];
                res.pos.x = vert.pos.x*mul.x, res.pos.y = -vert.pos.y*mul.y;
                res.pos.x -= 1.0f, res.pos.y += 1.0f;
                res.uv.x = vert.uv.x, res.uv.y = vert.uv.y;
                res.col = toRGBA(vert.col);
            }
        }
        buffer.memcpy(vbo, [buf = std::move(tmp)](auto&& call) {
            call(buf.data());
        });
    }

    const auto vert = calcVertex<VertInfo, VertOut, Empty, vertShader>(buffer, vbo, nullptr);
    const auto vboBase = DataPtr<VertexInfo<VertOut>>{ vert };
    auto vertBufferOffset = 0;

    auto ibo = buffer.allocBuffer<uvec3>(drawData->TotalIdxCount / 3);
    {
        std::vector<uvec3> tmp(ibo.size());
        for (auto i = 0; i < drawData->CmdListsCount; ++i){
            auto&& idxBuf = drawData->CmdLists[i]->IdxBuffer;
            for (auto j = 0; j < tmp.size(); ++j)
                tmp[j] = { idxBuf[j * 3],idxBuf[j * 3 + 1],idxBuf[j * 3 + 2] };
            buffer.memcpy(ibo, [buf = std::move(tmp)](auto&& call) {
                call(buf.data());
            });
        }
    }

    const auto iboBase = DataPtr<uvec3>{ ibo };
    auto idxBufferOffset = 0;

    for (auto i = 0; i < drawData->CmdListsCount; ++i) {
        const auto cmdList = drawData->CmdLists[i];

        for (auto j = 0; j < cmdList->CmdBuffer.size(); ++j) {
            const auto& pcmd = cmdList->CmdBuffer[j];
            if (pcmd.UserCallback)
                throw std::logic_error("Software renderer doesn't support user call back.");
            const vec4 scissor = { pcmd.ClipRect.x,pcmd.ClipRect.z, pcmd.ClipRect.y, pcmd.ClipRect.w };
            const auto faceCount = pcmd.ElemCount / 3;
            const auto index = makeIndexDescriptor<SeparateTrianglesWithIndex>(faceCount,
                ibo, faceCount);
            TriangleRenderingHistory history;
            history.reset(faceCount, 65536U);
            renderTriangles<decltype(index), VertOut, BuiltinSamplerGPU<float>,
                BuiltinRenderTargetGPU<RGBA8>, clipShader, fragShader>(buffer, 
                    vboBase + vertBufferOffset, index, uni,frameBuffer, renderTarget.size(), 0.5f, 1.5f, 
                    history, scissor, CullFace::None);
            idxBufferOffset += faceCount;
        }

        vertBufferOffset += cmdList->VtxBuffer.size();
    }
}

void SoftwareRenderer::init(Stream& resLoader) {
    //create font texture
    unsigned char* pixels;
    int width, height;
    ImGui::GetIO().Fonts->GetTexDataAsAlpha8(&pixels, &width, &height); 
    mTexture = std::make_unique<BuiltinArray<float>>(uvec2{width,height});
    const auto size = width * height;
    PinnedBuffer<float> floatPixels(size);
    for (auto i = 0; i < size; ++i)
        floatPixels[i] = pixels[i] * int2Float;
    checkError(cudaMemcpyToArrayAsync(mTexture->get(),0,0,floatPixels.get(),
        size*sizeof(float),cudaMemcpyHostToDevice, resLoader.get()));
    mSampler = std::make_unique<BuiltinSampler<float>>(mTexture->get());
    resLoader.sync();
}

void SoftwareRenderer::uninit() {
    mSampler.reset();
    mTexture.reset();
}

void SoftwareRenderer::render(CommandBuffer& buffer,BuiltinRenderTarget<RGBA8>& renderTarget) {
    ImGui::Render();
    renderDrawLists(ImGui::GetDrawData(), buffer, renderTarget);
}
