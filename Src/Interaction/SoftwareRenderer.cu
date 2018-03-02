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

CUDAINLINE vec4 toRGBA(const unsigned int col) {
    return vec4{ col >> 24,(col >> 16) & 0xff,(col >> 8) & 0xff,col & 0xff }*int2Float;
}

CUDAINLINE void vertShader(ImDrawVert in, const Empty&, vec3& cpos, 
    VertOut& out) {
    cpos = { in.pos.x,in.pos.y,1.0f };
    out.get<VertOutAttr::TexCoord>() = { in.uv.x,in.uv.y };
    out.get<VertOutAttr::Color>() = toRGBA(in.col);
}

CUDAINLINE bool clipShader(unsigned int, vec3&, vec3&, vec3&, const BuiltinSamplerGPU<A8>&) {
    return true;
}

CUDAINLINE void fragShader(unsigned int, ivec2 uv, float, const VertOut& in, const VertOut&, const VertOut&,
    const BuiltinSamplerGPU<A8>& texture, BuiltinRenderTargetGPU<RGBA8>& fbo) {
    const auto src = in.get<VertOutAttr::Color>()*(texture.get(in.get<VertOutAttr::TexCoord>())*int2Float);
    const auto dst = vec4(fbo.get(uv))*int2Float;
    const auto alpha = src.a, invAlpha = 1.0f - alpha;
    const auto col = vec3(src)*alpha + vec3(dst) * invAlpha;
    const auto res = clamp(vec4{ col,alpha*invAlpha }, 0.0f, 1.0f);
    fbo.set(uv, RGBA8{res*255.0f});
}

void SoftwareRenderer::renderDrawLists(ImDrawData* drawData,CommandBuffer& buffer,
    BuiltinRenderTarget<RGBA8>& renderTarget) {
    auto&& io = ImGui::GetIO();
    const int fbw = io.DisplaySize.x * io.DisplayFramebufferScale.x;
    const int fbh = io.DisplaySize.y * io.DisplayFramebufferScale.y;
    if (fbw == 0 || fbh == 0)return;
    drawData->ScaleClipRects(io.DisplayFramebufferScale);

    auto uni=buffer.allocConstant<BuiltinSamplerGPU<A8>>();
    buffer.memcpy(uni, [this](auto&& call) {
        const auto data = mSampler->toSampler();
        call(&data);
    });

    auto frameBuffer = buffer.allocConstant<BuiltinRenderTargetGPU<RGBA8>>();
    buffer.memcpy(frameBuffer,[rt=renderTarget.toTarget()](auto&& call) {
        call(&rt);
    });

    for (auto i = 0; i < drawData->CmdListsCount; ++i) {
        const auto cmdList = drawData->CmdLists[i];
        auto idxBufferOffset = 0;

        auto vbo = buffer.allocBuffer<ImDrawVert>(cmdList->VtxBuffer.size());
        {
            const auto mul = 2.0f / static_cast<vec2>(renderTarget.size());
            std::vector<ImDrawVert> tmp(cmdList->VtxBuffer.begin(), cmdList->VtxBuffer.end());
            for (auto& vert : tmp) {
                vert.pos.x *= mul.x, vert.pos.y *= -mul.y;
                vert.pos.x -= 1.0f, vert.pos.y += 1.0f;
            }
            buffer.memcpy(vbo, [buf = std::move(tmp)](auto&& call) {
                call(buf.data());
            });
        }

        auto vert = calcVertex<ImDrawVert, VertOut, Empty, vertShader>(buffer, vbo, nullptr);

        auto&& idxBuf = cmdList->IdxBuffer;
        auto ibo = buffer.allocBuffer<uvec3>(idxBuf.size() / 3);
        {
            std::vector<uvec3> tmp(ibo.size());
            for (auto j = 0; j < tmp.size(); ++j)
                tmp[j] = { idxBuf[j * 3],idxBuf[j * 3 + 1],idxBuf[j * 3 + 2] };
            buffer.memcpy(ibo, [buf=std::move(tmp)](auto&& call) {
                call(buf.data());
            });
        }
        const auto ptrBase = DataPtr<uvec3>{ ibo };

        for (auto j = 0; j < cmdList->CmdBuffer.size(); ++j) {
            const auto& pcmd = cmdList->CmdBuffer[j];
            if (pcmd.UserCallback)
                throw std::logic_error("Software renderer doesn't support user call back.");
            const vec4 scissor = { pcmd.ClipRect.x,pcmd.ClipRect.z,fbh - pcmd.ClipRect.w,fbh - pcmd.ClipRect.y };
            const auto ptr = ptrBase + idxBufferOffset;
            const auto faceCount = pcmd.ElemCount / 3;
            const auto index = makeIndexDescriptor<SeparateTrianglesWithIndex>(faceCount,
                ibo, faceCount);
            TriangleRenderingHistory history;
            history.reset(faceCount, 65536U);
            renderTriangles<decltype(index), VertOut, BuiltinSamplerGPU<A8>,
                BuiltinRenderTargetGPU<RGBA8>, clipShader, fragShader>(buffer, vert, index, uni,
                    frameBuffer, renderTarget.size(), 0.5f, 1.5f, history, scissor, CullFace::None);
            idxBufferOffset += faceCount;
        }
    }
}

void SoftwareRenderer::init(Stream& resLoader) {
    //create font texture
    unsigned char* pixels;
    int width, height;
    ImGui::GetIO().Fonts->GetTexDataAsAlpha8(&pixels, &width, &height); 
    mTexture = std::make_unique<BuiltinArray<A8>>(uvec2{width,height});
    checkError(cudaMemcpyToArrayAsync(mTexture->get(),0,0,pixels,width*height,
        cudaMemcpyHostToDevice, resLoader.get()));
    mSampler = std::make_unique<BuiltinSampler<A8>>(mTexture->get(), cudaAddressModeWrap,vec4{},
        cudaFilterModePoint);
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
