#include <Interaction/SoftwareRenderer.hpp>
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <Base/CompileEnd.hpp>
#include <ScanLineRenderer/TriangleRasterizer.hpp>
#include <Base/DataSet.hpp>
#include <ScanLineRenderer/IndexDescriptor.hpp>
#include <ScanLineRenderer/Buffer2D.hpp>

enum class VertOutAttr {
    TexCoord,
    Color
};

using VertOut = Args<VAR(VertOutAttr::TexCoord, vec2), VAR(VertOutAttr::Color, RGBA)>;

struct VertInfo final {
    ALIGN vec2 pos;
    ALIGN vec2 uv;
    ALIGN unsigned int col;
};

constexpr auto int2Float = 1.0f / 256.0f;

CUDAINLINE vec4 toRGBA(const unsigned int col) {
    return vec4{col & 0xff, (col >> 8) & 0xff, (col >> 16) & 0xff, col >> 24} * int2Float;
}

CUDAINLINE void vertShader(VertInfo in, const Empty&, vec3& cpos,
    VertOut& out) {
    cpos = {in.pos.x, in.pos.y, 1.0f};
    out.get<VertOutAttr::TexCoord>() = {in.uv.x, in.uv.y};
    out.get<VertOutAttr::Color>() = toRGBA(in.col);
}

struct FrameBufferInfo {
    BuiltinRenderTargetRef<RGBA8> color;
};

CUDAINLINE bool clipShader(unsigned int, vec3&, vec3&, vec3&, const BuiltinSamplerRef<float>&) {
    return true;
}

CUDAINLINE void colorShade(unsigned int id, ivec2 uv, float, const VertOut& in, const VertOut&, const VertOut&,
    const BuiltinSamplerRef<float>& texture, FrameBufferInfo& fbo) {
    const auto texAlpha = texture.get(in.get<VertOutAttr::TexCoord>());
    auto src = in.get<VertOutAttr::Color>();
    src.a *= texAlpha;
    const auto dst = vec4(fbo.color.get(uv)) * int2Float;
    const auto alpha = src.a, invAlpha = 1.0f - alpha;
    const auto col = vec3(src) * alpha + vec3(dst) * invAlpha;
    fbo.color.set(uv, RGBA8{clamp(col, 0.0f, 1.0f) * 255.0f, 255});
}

static bool isOverlap(vec2 a1, const vec2 a2, vec2 a3, vec2 b1, vec2 b2, vec2 b3) {
    const auto edgeFunction=[](const vec2 a,const vec2 b,const vec2 c) {
        return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
    };
    if (edgeFunction(a1, a2, a3) < 0.0f)std::swap(a1, a3);
    const auto test = [&](const vec2 a, const vec2 b, const vec2 c) {
        return edgeFunction(a, b, c) <= 0.0f;
    };
    const auto out=[&](const vec2 a,const vec2 b) {
        return test(a, b, b1) && test(a, b, b2) && test(a, b, b3);
    };
    return !(out(a1, a2) || out(a2, a3) || out(a3, a1));
}

void SoftwareRenderer::render(CommandBuffer& buffer, BuiltinRenderTarget<RGBA8>& renderTarget) {
    const auto drawData = ImGui::GetDrawData();
    #ifdef CAMERA_DEBUG
    if(!drawData->Valid)throw std::logic_error("This draw data is invalid.");
    #endif
    auto&& io = ImGui::GetIO();
    const int fbw = io.DisplaySize.x * io.DisplayFramebufferScale.x;
    const int fbh = io.DisplaySize.y * io.DisplayFramebufferScale.y;
    if (fbw == 0 || fbh == 0)return;
    drawData->ScaleClipRects(io.DisplayFramebufferScale);

    auto uni = buffer.allocConstant<BuiltinSamplerRef<float>>();
    buffer.memcpy(uni, [this](auto&& call) {
        const auto data = mSampler->toSampler();
        call(&data);
    });

    auto frameBuffer = buffer.allocConstant<FrameBufferInfo>();
    buffer.memcpy(frameBuffer, [rt = renderTarget.toTarget()](auto&& call) {
            FrameBufferInfo info;
            info.color = rt;
            call(&info);
        });

    auto vbo = buffer.allocBuffer<VertInfo>(drawData->TotalVtxCount);
    std::vector<VertInfo> vertData(vbo.size());
    {
        const auto mul = 2.0f / static_cast<vec2>(renderTarget.size());
        auto idx = 0U;
        for (auto i = 0; i < drawData->CmdListsCount; ++i) {
            for (const auto& vert : drawData->CmdLists[i]->VtxBuffer) {
                auto&& res = vertData[idx++];
                res.pos.x = vert.pos.x * mul.x, res.pos.y = -vert.pos.y * mul.y;
                res.pos.x -= 1.0f, res.pos.y += 1.0f;
                res.uv.x = vert.uv.x, res.uv.y = vert.uv.y;
                res.col = vert.col;
            }
        }
        buffer.memcpy(vbo, [buf = vertData](auto&& call) {
            call(buf.data());
        });
    }

    const auto vert = calcVertex<VertInfo, VertOut, Empty, vertShader>(buffer, vbo, nullptr);
    const auto vertBase = DataPtr<VertexInfo<VertOut>>{vert};
    auto vertBufferOffset = 0;

    auto ibo = buffer.allocBuffer<uvec3>(drawData->TotalIdxCount / 3);
    std::vector<uvec3> indexData(ibo.size());
    {
        auto idx = 0;
        for (auto i = 0; i < drawData->CmdListsCount; ++i) {
            auto&& idxBuf = drawData->CmdLists[i]->IdxBuffer;
            const auto idxSiz = idxBuf.size() / 3;
            for (auto j = 0; j < idxSiz; ++j)
                indexData[idx++] = { idxBuf[j * 3], idxBuf[j * 3 + 1], idxBuf[j * 3 + 2] };
        }
        buffer.memcpy(ibo, [buf = indexData](auto&& call) {
            call(buf.data());
        });
    }

    const auto testOverlap = [&vertData, &indexData](auto i, auto j, auto vertOffset, auto indexOffset) {
        const auto ii = indexData[i + indexOffset], ij = indexData[j + indexOffset];
        const auto base = vertData.data() + vertOffset;
        return isOverlap(base[ii[0]].pos, base[ii[1]].pos, base[ii[2]].pos,
            base[ij[0]].pos, base[ij[1]].pos, base[ij[2]].pos);
    };

    const auto iboBase = DataPtr<uvec3>{ibo};
    auto idxBufferOffset = 0;

    for (auto i = 0; i < drawData->CmdListsCount; ++i) {
        const auto cmdList = drawData->CmdLists[i];
        const auto vertPtr= vertBase + vertBufferOffset;

        for (auto j = 0; j < cmdList->CmdBuffer.size(); ++j) {
            const auto& cmd = cmdList->CmdBuffer[j];

            if (cmd.UserCallback)
                throw std::logic_error("Software renderer doesn't support user call back.");
            const vec4 scissor = {cmd.ClipRect.x, cmd.ClipRect.z, cmd.ClipRect.y, cmd.ClipRect.w};
            const auto faceCount = cmd.ElemCount / 3;
            const auto render = [&](auto base, auto size) {
                const auto idxPtr = iboBase + (idxBufferOffset + base);
                const auto index = makeIndexDescriptor<SeparateTrianglesWithIndex>(size, idxPtr.get());
                TriangleRenderingHistory history;
                history.reset(size, 65536U);
                renderTriangles<decltype(index), VertOut, BuiltinSamplerRef<float>,
                    FrameBufferInfo, clipShader, colorShade>(buffer,vertPtr, index, uni,frameBuffer,
                    renderTarget.size(), 0.5f, 1.5f, history, scissor, CullFace::None);
            };
            auto current = 0U;
            for (auto k = 0U; k < faceCount; ++k) {
                auto flag = false;
                for (auto l = current; l < k; ++l)
                    if (testOverlap(k, l, vertBufferOffset, idxBufferOffset)) {
                        flag = true;
                        break;
                    }

                if(flag){
                    render(current, k - current);
                    current = k;
                }
            }

            render(current, faceCount - current);

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
    mTexture = std::make_unique<BuiltinArray<float>>(uvec2{width, height});
    const auto size = width * height;
    PinnedBuffer<float> floatPixels(size);
    for (auto i = 0; i < size; ++i)
        floatPixels[i] = pixels[i] * int2Float;
    checkError(cudaMemcpyToArrayAsync(mTexture->get(), 0, 0, floatPixels.get(),
        size * sizeof(float), cudaMemcpyHostToDevice, resLoader.get()));
    mSampler = std::make_unique<BuiltinSampler<float>>(mTexture->get());
    ImGui::GetIO().Fonts->SetTexID(mSampler.get());
    resLoader.sync();
}

void SoftwareRenderer::uninit() {
    mSampler.reset();
    mTexture.reset();
}
