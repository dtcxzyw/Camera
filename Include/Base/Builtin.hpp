#pragma once
#include <Base/Math.hpp>
#include <Base/Common.hpp>

#include <Base/CompileBegin.hpp>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_indirect_functions.h>
#include <cuda_surface_types.h>
#include <surface_indirect_functions.h>
#include <Base/CompileEnd.hpp>

#include <Base/Pipeline.hpp>
#include <Base/DispatchSystem.hpp>

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

    DataViewer<T> download(CommandBuffer& buffer) const {
        auto res = allocBuffer<T>(mSize.x * mSize.y);
        buffer.pushOperator([=](Stream& stream) {
            checkError(cudaMemcpyFromArrayAsync(res.begin(), mArray, 0, 0, res.size() * sizeof(T)
                                                , cudaMemcpyDefault, stream.getID()));
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

template <typename T>
class BuiltinMipmapedArray final : Uncopyable {
private:
    cudaMipmappedArray_t mArray;
    using Type = typename Rename<T>::Type;
    uvec2 mSize;

    void genMipmaps(const void* src, const unsigned int level, Stream& stream) {
        cudaArray_t srcArray;
        {
            cudaMemcpy3DParms parms;
            parms.extent = {mSize.x, mSize.y, 0};
            parms.kind = cudaMemcpyHostToDevice;
            parms.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), mSize.x * sizeof(T), mSize.x, mSize.y);
            checkError(cudaGetMipmappedArrayLevel(&srcArray, mArray, 0));
            parms.dstArray = srcArray;
            checkError(cudaMemcpy3D(&parms));
        }
        auto size = mSize;
        for (unsigned int i = 1; i < level; ++i) {
            size = max(size / 2U, uvec2{1, 1});
            cudaArray_t dstArray;
            checkError(cudaGetMipmappedArrayLevel(&dstArray, mArray, i));
            downSample<T>(srcArray, dstArray, size, stream);
            srcArray = dstArray;
        }
        stream.sync();
    }

public:
    BuiltinMipmapedArray(const void* src, const uvec2 size, Stream& stream,
                         const int flags = cudaArrayDefault, unsigned int level = 0) : mSize(size) {
        auto desc = cudaCreateChannelDesc<Type>();
        const auto maxLevel = calcMaxMipmapLevel(size);
        if (level == 0)level = maxLevel;
        else level = std::max(level, maxLevel);
        checkError(cudaMallocMipmappedArray(&mArray, &desc,
                                            make_cudaExtent(size.x, size.y, 0), level, flags));
        genMipmaps(src, level, stream);
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
class BuiltinSamplerGPU final {
public:
    using Type = typename Rename<T>::Type;
    CUDAINLINE BuiltinSamplerGPU(): mTexture(0) {}
    explicit BuiltinSamplerGPU(const cudaTextureObject_t texture): mTexture(texture) {}
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

    CUDAINLINE T getGrad(vec2 p, vec2 ddx, vec2 ddy) const {
        T res;
        tex2DGrad<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, 
            *reinterpret_cast<float2*>(&ddx),*reinterpret_cast<float2*>(&ddy));
        return res;
    }

    CUDAINLINE T getCubeMap(vec3 p) const {
        T res;
        texCubemap<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, p.z);
        return res;
    }

    CUDAINLINE T getCubeMapGrad(vec3 p,vec4 ddx,vec4 ddy) const {
        T res;
        texCubemapGrad<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, p.z,
            *reinterpret_cast<float4*>(&ddx),*reinterpret_cast<float4*>(&ddy));
        return res;
    }

private:
    cudaTextureObject_t mTexture;
};

//The trick comes from https://learnopengl.com/#!PBR/IBL/Diffuse-irradiance.
CUDAINLINE vec2 calcHDRUV(const vec3 p) {
    return {atan(p.z, p.x) * 0.1591f + 0.5f, asin(p.y) * 0.3183f + 0.5f};
}

template <typename T>
class BuiltinSampler final : Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaTextureObject_t mTexture;
public:
    explicit BuiltinSampler(const cudaArray_t array,
                   const cudaTextureAddressMode am = cudaAddressModeWrap,
                   const vec4 borderColor = {}, const cudaTextureFilterMode fm = cudaFilterModeLinear,
                   const unsigned int maxAnisotropy = 0,
                   const bool sRGB = false) {
        cudaResourceDesc RD;
        RD.res.array.array = array;
        RD.resType = cudaResourceTypeArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.addressMode[0] = TD.addressMode[1] = am;
        *reinterpret_cast<vec4*>(TD.borderColor) = borderColor;
        TD.filterMode = fm;
        TD.maxAnisotropy = maxAnisotropy;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        TD.sRGB = sRGB;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &TD, nullptr));
    }

    explicit BuiltinSampler(BuiltinMipmapedArray<T>& array,
                   const cudaTextureAddressMode am = cudaAddressModeWrap, const vec4 borderColor = {},
                   const cudaTextureFilterMode fm = cudaFilterModeLinear, const unsigned int maxAnisotropy = 0,
                   const bool sRGB = false) {
        cudaResourceDesc RD;
        RD.res.mipmap.mipmap = array.get();
        RD.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.addressMode[0] = TD.addressMode[1] = am;
        *reinterpret_cast<vec4*>(TD.borderColor) = borderColor;
        TD.filterMode = fm;
        TD.maxAnisotropy = maxAnisotropy;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        TD.sRGB = sRGB;
        auto size = array.size();
        TD.maxMipmapLevelClamp = calcMaxMipmapLevel(size.x, size.y);
        TD.minMipmapLevelClamp = 1.0f;
        TD.mipmapFilterMode = fm;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &TD, nullptr));
    }

    auto toSampler() const {
        return BuiltinSamplerGPU<T>{ mTexture };
    }

    ~BuiltinSampler() {
        checkError(cudaDestroyTextureObject(mTexture));
    }
};

template <typename T>
class BuiltinRenderTargetGPU final {
public:
    using Type = typename Rename<T>::Type;
    CUDAINLINE BuiltinRenderTargetGPU(): mTarget(0) {};
    explicit BuiltinRenderTargetGPU(const cudaSurfaceObject_t target) : mTarget(target) {}
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
    GLOBAL void clear(BuiltinRenderTargetGPU<T> rt, T val) {
        uvec2 p = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
        rt.set(p, val);
    }
}

template <typename T>
class BuiltinRenderTarget final : Uncopyable {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    cudaSurfaceObject_t mTarget;
    uvec2 mSize;
public:
    BuiltinRenderTarget(const cudaArray_t array, const uvec2 size) : mArray(array), mSize(size) {
        cudaResourceDesc RD;
        RD.res.array.array = mArray;
        RD.resType = cudaResourceTypeArray;
        checkError(cudaCreateSurfaceObject(&mTarget, &RD));
    }

    explicit BuiltinRenderTarget(BuiltinArray<T>& array): BuiltinRenderTarget(array.get(), array.size()) {}

    auto toTarget() const {
        return BuiltinRenderTargetGPU<T>{ mTarget };
    }

    void clear(CommandBuffer& buffer, T val) {
        const uint mul = sqrt(getEnvironment().getProp().maxThreadsPerBlock);
        dim3 grid(calcSize(mSize.x, mul), calcSize(mSize.y, mul));
        dim3 block(mul, mul);
        buffer.runKernelDim(Impl::clear<T>, grid, block, toTarget(), val);
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
    explicit BuiltinCubeMap(const size_t size) {
        auto desc = cudaCreateChannelDesc<Type>();
        const cudaExtent extent{size, size, 6};
        checkError(cudaMalloc3DArray(&mArray, &desc, extent,cudaArrayCubemap));
        cudaResourceDesc RD;
        RD.resType = cudaResourceTypeArray;
        RD.res.array.array = mArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
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

    BuiltinSamplerGPU<T> toSampler() const {
        return mTexture;
    }
};

//TODO:complete mipmaped cube map
template <typename T>
class BuiltinMipmapedCubeMap final : Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaMipmappedArray_t mArray;
    cudaTextureObject_t mTexture;
public:
    explicit BuiltinMipmapedCubeMap(const size_t size) {
        auto desc = cudaCreateChannelDesc<Type>();
        const cudaExtent extent{ size, size, 6 };
        checkError(cudaMallocMipmappedArray(&mArray, &desc, extent, cudaArrayCubemap));
        cudaResourceDesc RD;
        RD.resType = cudaResourceTypeMipmappedArray;
        RD.res.mipmap.mipmap = mArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.filterMode = cudaFilterModeLinear;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &TD, nullptr));
    }

    ~BuiltinMipmapedCubeMap() {
        checkError(cudaDestroyTextureObject(mTexture));
        checkError(cudaFreeMipmappedArray(mArray));
    }

    cudaMipmappedArray_t get() const {
        return mArray;
    }

    BuiltinSamplerGPU<T> toSampler() const {
        return mTexture;
    }
};

namespace Impl {
    template <typename T>
    GLOBAL void downSample(BuiltinSamplerGPU<T> src, BuiltinRenderTargetGPU<T> rt) {
        uvec2 p{ blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
        const uvec2 base = { p.x * 2, p.y * 2 };
        T val = {};
        for (auto i = 0; i < 2; ++i)
            for (auto j = 0; j < 2; ++j)
                val += src.get(base + uvec2{ i, j });
        rt.set(p, val * 0.25f);
    }
}

template <typename T>
void downSample(cudaArray_t src, cudaArray_t dst, uvec2 size, Stream& stream) {
    BuiltinSampler<T> sampler(src);
    BuiltinRenderTarget<T> RT(dst, size);
    const uint mul = sqrt(getEnvironment().getProp().maxThreadsPerBlock);
    dim3 grid(calcSize(size.x, mul), calcSize(size.y, mul));
    dim3 block(mul, mul);
    stream.runDim(Impl::downSample<T>, grid, block, sampler.toSampler(), RT.toTarget());
}

