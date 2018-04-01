#include <Base/Config.hpp>
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

constexpr auto int2Float = 1.0f / 255.0f;

CUDAINLINE vec4 toRGBA(const unsigned int col) {
    return vec4{col & 0xff, (col >> 8) & 0xff, (col >> 16) & 0xff, col >> 24} * int2Float;
}

CUDAINLINE void vertShader(VertInfo in, const Empty&, vec3& cpos, VertOut& out) {
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

static bool isOverlap(vec2 a1, const vec2 a2, vec2 a3, const vec2 b1, const vec2 b2, const vec2 b3) {
    const auto edgeFunction = [](const vec2 a,const vec2 b,const vec2 c) {
        return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
    };
    if (edgeFunction(a1, a2, a3) < 0.0f)std::swap(a1, a3);
    const auto test = [&](const vec2 a, const vec2 b, const vec2 c) {
        return edgeFunction(a, b, c) <= 1e-6f;
    };
    const auto out = [&](const vec2 a,const vec2 b) {
        return test(a, b, b1) && test(a, b, b2) && test(a, b, b3);
    };
    return !(out(a1, a2) || out(a2, a3) || out(a3, a1));
}

static std::vector<std::pair<size_t, size_t>> sortPrimitives(const VertInfo* vert,
    const uvec3* index, const size_t idxCount) {
    const auto testOverlap = [vert](const uvec3& a,const uvec3& b) {
        return isOverlap(vert[a.x].pos, vert[a.y].pos, vert[a.z].pos,
            vert[b.x].pos, vert[b.y].pos, vert[b.z].pos);
    };
    const auto canUnion = [index,&testOverlap](const uvec3& cur,const std::pair<size_t,size_t>& set) {
        for (auto i = 0; i < set.second; ++i)
            if (testOverlap(index[set.first + i], cur))
                return false;
        return true;
    };
    std::vector<std::pair<size_t, size_t>> res;
    for (size_t i = 0; i < idxCount; ++i) {
        const auto cur = index[i];
        if (!res.empty() && canUnion(cur, res.back()))++res.back().second;
        else res.emplace_back(i, 1);
    }
    return res;
}

void SoftwareRenderer::render(CommandBuffer& buffer, BuiltinRenderTarget<RGBA8>& renderTarget) {
    const auto drawData = ImGui::GetDrawData();
    #ifdef CAMERA_DEBUG
    if (!drawData->Valid)throw std::logic_error("This draw data is invalid.");
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

    std::vector<VertInfo> vertData(drawData->TotalVtxCount);
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
    }

    struct DrawCommand final {
        size_t vertOffset, idxOffset, idxCount;
        vec4 scissor;
    };

    std::vector<DrawCommand> drawCmd;
    std::vector<uvec3> idxData(drawData->TotalIdxCount / 3);
    {
        size_t vertOffset = 0;
        size_t idxOffset = 0;
        for (auto i = 0; i < drawData->CmdListsCount; ++i) {
            auto&& list = drawData->CmdLists[i];
            auto&& idxBuf = list->IdxBuffer;
            const auto idxSize = idxBuf.size() / 3;
            for (auto j = 0; j < idxSize; ++j)
                idxData[idxOffset + j] = { idxBuf[j * 3], idxBuf[j * 3 + 1], idxBuf[j * 3 + 2] };
            auto&& cmdBuffer = list->CmdBuffer;
            for (auto j = 0; j < cmdBuffer.size(); ++j) {
                auto&& cmd = cmdBuffer[j];
                if (cmd.UserCallback)
                    throw std::logic_error("Software renderer doesn't support user call back.");
                const auto cnt = cmd.ElemCount / 3;
                auto drawCalls = sortPrimitives(vertData.data() + vertOffset, idxData.data() + idxOffset, cnt);
                for (auto&& drawCall : drawCalls) {
                    DrawCommand info;
                    info.vertOffset = vertOffset;
                    info.idxOffset = idxOffset + drawCall.first;
                    info.idxCount = drawCall.second;
                    info.scissor = {cmd.ClipRect.x, cmd.ClipRect.z, cmd.ClipRect.y, cmd.ClipRect.w};
                    drawCmd.emplace_back(info);
                }
                idxOffset += cnt;
            }
            vertOffset += list->VtxBuffer.size();
        }
    }

    auto vbo = buffer.allocBuffer<VertInfo>(drawData->TotalVtxCount);
    buffer.memcpy(vbo, [buf = std::move(vertData)](auto&& call) {
        call(buf.data());
    });
    const auto vert = calcVertex<VertInfo, VertOut, Empty, vertShader>(buffer, vbo, nullptr);
    const auto vertBase = DataPtr<VertexInfo<VertOut>>{vert};

    auto ibo = buffer.allocBuffer<uvec3>(idxData.size());
    buffer.memcpy(ibo, [buf = std::move(idxData)](auto&& call) {
        call(buf.data());
    });
    const auto iboBase = DataPtr<uvec3>{ibo};

    #ifdef CAMERA_SOFTWARE_RENDERER_COUNT_DRAWCALL
    printf("draw call %u\n", static_cast<unsigned int>(drawCmd.size()));
    #endif

    for(auto&& cmd:drawCmd) {
        const auto vertPtr = vertBase + cmd.vertOffset;
        const auto idxPtr = iboBase + cmd.idxOffset;
        const auto index = makeIndexDescriptor<SeparateTrianglesWithIndex>(cmd.idxCount, idxPtr.get());
        TriangleRenderingHistory history;
        history.reset(cmd.idxCount, 65536U);
        renderTriangles<decltype(index), VertOut, BuiltinSamplerRef<float>,
            FrameBufferInfo, clipShader, colorShade>(buffer, vertPtr, index, uni, frameBuffer,
                renderTarget.size(), 0.5f, 1.5f, history, cmd.scissor, CullFace::None);
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
