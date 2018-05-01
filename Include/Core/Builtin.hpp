#pragma once
#include <Math/Math.hpp>
#include <Core/Common.hpp>
#include <Core/CompileBegin.hpp>
#include <cuda_texture_types.h>
#include <texture_indirect_functions.h>
#include <cuda_surface_types.h>
#include <surface_indirect_functions.h>
#include <Core/CompileEnd.hpp>
#include <Core/DispatchSystem.hpp>
#include <Core/CompileBegin.hpp>
#include <glm/gtc/round.hpp>
#include <Core/CompileEnd.hpp>

template <typename T>
struct Rename final {
    using Type = T;
};

template <>
struct Rename<RGBA> final {
    using Type = float4;
};

template <>
struct Rename<UV> final {
    using Type = float2;
};

template <>
struct Rename<RGBA8> final {
    using Type = uchar4;
};

template <typename T>
class BuiltinArray final : Uncopyable {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    uvec2 mSize;
public:
    explicit BuiltinArray(const uvec2 size, const int flag = cudaArrayDefault) : mSize(size) {
        auto desc = cudaCreateChannelDesc<Type>();
        checkError(cudaMallocArray(&mArray, &desc, size.x, size.y, flag));
    }

    ~BuiltinArray() {
        checkError(cudaFreeArray(mArray));
    }

    MemorySpan<T> download(CommandBuffer& buffer) const {
        MemorySpan<T> res(mSize.x * mSize.y);
        buffer.pushOperator([=](Stream& stream) {
            checkError(cudaMemcpyFromArrayAsync(res.begin(), mArray, 0, 0, res.size() * sizeof(T)
                , cudaMemcpyDefault, stream.get()));
        });
        return res;
    }

    uvec2 size() const {
        return mSize;
    }

    cudaArray_t get() const {
        return mArray;
    }
};

inline auto calcMaxMipmapLevel(const uvec2 size) {
    return static_cast<unsigned int>(floor(std::log2(std::max(size.x, size.y)))) + 1U;
}

template <typename T>
void downSample(cudaArray_t src, cudaArray_t dst, uvec2 size, Stream& stream);

template<typename T>
void scaleArray(const BuiltinArray<T>& src, cudaArray_t dstArray, uvec2 dstSize,
    Stream& stream);

template <typename T>
class BuiltinMipmapedArray final : Uncopyable {
private:
    cudaMipmappedArray_t mArray;
    using Type = typename Rename<T>::Type;
    uvec2 mSize;
    unsigned int mLevel;

    void genMipmaps(const BuiltinArray<T>& src, Stream& stream) {
        cudaArray_t srcArray;
        checkError(cudaGetMipmappedArrayLevel(&srcArray, mArray, 0));
        scaleArray(src, srcArray, mSize, stream);
        auto size = mSize;
        for (unsigned int i = 1; i < mLevel; ++i) {
            size /= 2U;
            cudaArray_t dstArray;
            checkError(cudaGetMipmappedArrayLevel(&dstArray, mArray, i));
            downSample<T>(srcArray, dstArray, size, stream);
            srcArray = dstArray;
        }
        stream.sync();
    }

public:
    BuiltinMipmapedArray(const BuiltinArray<T>& src, Stream& stream,
        const int flags = cudaArrayDefault, const unsigned int level = 0) :mLevel(level) {
        const auto desc = cudaCreateChannelDesc<Type>();
        const auto size = src.size();
        const auto maxLevel = calcMaxMipmapLevel(size);
        if (mLevel == 0)mLevel = maxLevel;
        else mLevel = std::min(mLevel, maxLevel);
        const auto length = glm::ceilPowerOfTwo(std::max(size.x, size.y));
        mSize = uvec2{ length };
        checkError(cudaMallocMipmappedArray(&mArray, &desc,
            make_cudaExtent(length, length, 0), level, flags));
        genMipmaps(src, stream);
    }

    ~BuiltinMipmapedArray() {
        checkError(cudaFreeMipmappedArray(mArray));
    }

    uvec2 size() const {
        return mSize;
    }

    cudaMipmappedArray_t get() const {
        return mArray;
    }
};

template <typename T>
class BuiltinSamplerRef final {
public:
    using Type = typename Rename<T>::Type;
    CUDAINLINE BuiltinSamplerRef(): mTexture(0) {}
    explicit BuiltinSamplerRef(const cudaTextureObject_t texture): mTexture(texture) {}
    CUDAINLINE T get(const vec2 p) const {
        T res;
        tex2D<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y);
        return res;
    }

    CUDAINLINE T getGather(const vec2 p, const int comp) const {
        T res;
        tex2Dgather<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, comp);
        return res;
    }

    CUDAINLINE T getGrad(const vec2 p, const vec2 ddx, const vec2 ddy) const {
        T res;
        tex2DGrad<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y,
            *reinterpret_cast<const float2*>(&ddx), *reinterpret_cast<const float2*>(&ddy));
        return res;
    }

    CUDAINLINE T getCubeMap(const Vector p) const {
        T res;
        texCubemap<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, p.z);
        return res;
    }

    CUDAINLINE T getCubeMapGrad(const Vector p, const vec4 ddx, const vec4 ddy) const {
        T res;
        texCubemapGrad<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, p.z,
            *reinterpret_cast<const float4*>(&ddx), *reinterpret_cast<const float4*>(&ddy));
        return res;
    }

private:
    cudaTextureObject_t mTexture;
};

template <typename T>
class BuiltinSampler final : Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaTextureObject_t mTexture;
public:
    explicit BuiltinSampler(const cudaArray_t array,
        const cudaTextureAddressMode am = cudaAddressModeWrap,
        const vec4 borderColor = {}, const cudaTextureFilterMode fm = cudaFilterModeLinear,
        const unsigned int maxAnisotropy = 0) {
        cudaResourceDesc RD;
        RD.res.array.array = array;
        RD.resType = cudaResourceTypeArray;
        cudaTextureDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.addressMode[0] = desc.addressMode[1] = am;
        *reinterpret_cast<RGBA*>(desc.borderColor) = borderColor;
        desc.filterMode = fm;
        desc.maxAnisotropy = maxAnisotropy;
        desc.normalizedCoords = 1;
        desc.readMode = cudaReadModeElementType;
        desc.sRGB = false;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &desc, nullptr));
    }

    explicit BuiltinSampler(BuiltinMipmapedArray<T>& array,
        const cudaTextureAddressMode am = cudaAddressModeWrap, const vec4 borderColor = {},
        const cudaTextureFilterMode fm = cudaFilterModeLinear, const unsigned int maxAnisotropy = 0,
        const bool sRGB = false) {
        cudaResourceDesc RD;
        RD.res.mipmap.mipmap = array.get();
        RD.resType = cudaResourceTypeMipmappedArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.addressMode[0] = TD.addressMode[1] = am;
        *reinterpret_cast<RGBA*>(TD.borderColor) = borderColor;
        TD.filterMode = fm;
        TD.maxAnisotropy = maxAnisotropy;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        TD.sRGB = sRGB;
        auto size = array.size();
        TD.maxMipmapLevelClamp = array.maxLevel();
        TD.minMipmapLevelClamp = 1.0f;
        TD.mipmapFilterMode = fm;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &TD, nullptr));
    }

    auto toSampler() const {
        return BuiltinSamplerRef<T>{mTexture};
    }

    ~BuiltinSampler() {
        checkError(cudaDestroyTextureObject(mTexture));
    }
};

template <typename T>
class BuiltinRenderTargetRef final {
public:
    using Type = typename Rename<T>::Type;
    CUDAINLINE BuiltinRenderTargetRef(): mTarget(0) {};
    explicit BuiltinRenderTargetRef(const cudaSurfaceObject_t target) : mTarget(target) {}
    CUDAINLINE T get(ivec2 p) const {
        auto res = surf2Dread<Type>(mTarget, p.x * sizeof(Type), p.y, cudaBoundaryModeClamp);
        return *reinterpret_cast<T*>(&res);
    }

    CUDAINLINE void set(ivec2 p, T v) {
        auto val = *reinterpret_cast<Type*>(&v);
        surf2Dwrite(val, mTarget, p.x * sizeof(Type), p.y, cudaBoundaryModeZero);
    }

private:
    cudaSurfaceObject_t mTarget;
};

namespace Impl {
    template <typename T>
    GLOBAL void clear(BuiltinRenderTargetRef<T> rt, T val) {
        uvec2 p = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
        rt.set(p, val);
    }
}

template <typename T>
class BuiltinRenderTarget final : Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaArray_t mArray;
    cudaSurfaceObject_t mTarget;
    uvec2 mSize;
public:
    BuiltinRenderTarget(const cudaArray_t array, const uvec2 size) : mArray(array), mTarget(0), 
        mSize(size) {
        cudaResourceDesc desc;
        desc.res.array.array = mArray;
        desc.resType = cudaResourceTypeArray;
        checkError(cudaCreateSurfaceObject(&mTarget, &desc));
    }

    explicit BuiltinRenderTarget(BuiltinArray<T>& array): BuiltinRenderTarget(array.get(), array.size()) {}

    auto toTarget() const {
        return BuiltinRenderTargetRef<T>{mTarget};
    }

    void clear(CommandBuffer& buffer, T val) {
        buffer.pushOperator([this,val](Id,ResourceManager&,Stream& stream) {
            const unsigned int mul = sqrt(stream.getMaxBlockSize());
            dim3 grid(calcBlockSize(mSize.x, mul), calcBlockSize(mSize.y, mul));
            dim3 block(mul, mul);
            stream.launchDim(Impl::clear<T>, grid, block, toTarget(), val);
        });
    }

    uvec2 size() const {
        return mSize;
    }

    cudaArray_t get() const {
        return mArray;
    }

    ~BuiltinRenderTarget() {
        checkError(cudaDestroySurfaceObject(mTarget));
    }
};

template <typename T>
class BuiltinCubeMap final : Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaArray_t mArray;
    cudaTextureObject_t mTexture;
public:
    explicit BuiltinCubeMap(const size_t size, const unsigned int maxAnisotropy = 0,
        const unsigned int flag = cudaArrayDefault) {
        auto desc = cudaCreateChannelDesc<Type>();
        const cudaExtent extent{size, size, 6};
        checkError(cudaMalloc3DArray(&mArray, &desc, extent, cudaArrayCubemap | flag));
        cudaResourceDesc RD;
        RD.resType = cudaResourceTypeArray;
        RD.res.array.array = mArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.maxAnisotropy = maxAnisotropy;
        TD.sRGB = false;
        TD.addressMode[0] = TD.addressMode[1] = TD.addressMode[2] = cudaAddressModeClamp;
        TD.filterMode = cudaFilterModeLinear;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &TD, nullptr));
    }

    ~BuiltinCubeMap() {
        checkError(cudaDestroyTextureObject(mTexture));
        checkError(cudaFreeArray(mArray));
    }

    cudaArray_t get() const {
        return mArray;
    }

    BuiltinSamplerRef<T> toSampler() const {
        return mTexture;
    }
};

namespace Impl {
    template <typename T>
    GLOBAL void downSample2(BuiltinRenderTargetRef<T> src, BuiltinRenderTargetRef<T> rt) {
        uvec2 p{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
        const uvec2 base = {p.x << 1, p.y << 1};
        T val = {};
        #pragma unroll
        for (auto i = 0; i < 2; ++i) {
            #pragma unroll
            for (auto j = 0; j < 2; ++j)
                val += src.get(base + uvec2{ i, j });
        }
        rt.set(p, val * 0.25f);
    }
}

template <typename T>
void downSample(cudaArray_t srcArray, cudaArray_t dstArray, uvec2 size, Stream& stream) {
    BuiltinRenderTarget<T> src(srcArray, size * 2U);
    BuiltinRenderTarget<T> dst(dstArray, size);
    const unsigned int mul = sqrt(stream.getMaxBlockSize());
    dim3 grid(calcBlockSize(size.x, mul), calcBlockSize(size.y, mul));
    dim3 block(mul, mul);
    stream.launchDim(Impl::downSample2<T>, grid, block, src.toTarget(), dst.toTarget());
}

namespace Impl {
    template <typename T>
    GLOBAL void upSample(BuiltinSamplerRef<T> src, BuiltinRenderTargetRef<T> rt, vec2 mul) {
        uvec2 p{ blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
        rt.set(p, src.get(vec2(p)*mul));
    }
}

template<typename T>
void scaleArray(const BuiltinArray<T>& srcArray, cudaArray_t dstArray, const uvec2 dstSize,
    Stream& stream) {
    BuiltinSampler<T> src(srcArray.get());
    BuiltinRenderTarget<T> dst(dstArray, dstSize);
    const unsigned int mul = sqrt(stream.getMaxBlockSize());
    dim3 grid(calcBlockSize(dstSize.x, mul), calcBlockSize(dstSize.y, mul));
    dim3 block(mul, mul);
    const auto invSize = 1.0f / vec2(dstSize);
    stream.launchDim(Impl::upSample<T>, grid, block, src.toSampler(), dst.toTarget(), invSize);
}

template<typename T>
auto makeConstantTexture(const T& val) {
    auto res = std::make_unique<BuiltinArray<T>>(uvec2{ 1,1 });
    checkError(cudaMemcpy(res.get(), &val, sizeof(T), cudaMemcpyHostToDevice));
    return res;
}
