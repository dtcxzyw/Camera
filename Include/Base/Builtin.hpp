#pragma once
#include "Common.hpp"
#define __CUDACC__
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_indirect_functions.h>
#include <cuda_surface_types.h>
#include <surface_indirect_functions.h>
#include <Base/Pipeline.hpp>
#include <Base/DispatchSystem.hpp>

template<typename T>
struct Rename final {
    using Type = T;
};

template<>
struct Rename<RGBA> final {
    using Type = float4;
};

template<>
struct Rename<UV> final {
    using Type = float2;
};

template<typename T>
class BuiltinArray final :Uncopyable {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    uvec2 mSize;
public:
    BuiltinArray(size_t width, size_t height) :mSize(width, height) {
        auto desc = cudaCreateChannelDesc<Type>();
        checkError(cudaMallocArray(&mArray, &desc, width, height));
    }
    ~BuiltinArray() {
        checkError(cudaFreeArray(mArray));
    }
    DataViewer<T> download(CommandBuffer& buffer) const {
        auto res = allocBuffer<T>(mSize.x*mSize.y);
        buffer.pushOperator([=](Stream& stream) {
            checkError(cudaMemcpyFromArrayAsync(res.begin(), mArray, 0, 0, res.size() * sizeof(T)
                , cudaMemcpyDefault, stream.getID()));
        });
        return res;
    }
    DataViewer<T> download(Stream& stream) const {
        auto res = allocBuffer<T>(mSize.x*mSize.y);
        checkError(cudaMemcpyFromArrayAsync(res.begin(), mArray, 0, 0, mSize.x*mSize.y * sizeof(T)
            , cudaMemcpyDefault, stream.getID()));
        return res;
    }
    uvec2 size() const {
        return mSize;
    }
    cudaArray_t get() const {
        return mArray;
    }
};


template<typename T>
class BuiltinSamplerGPU final {
public:
    using Type = typename Rename<T>::Type;
    CUDA BuiltinSamplerGPU() = default;
    BuiltinSamplerGPU(cudaTextureObject_t texture):mTexture(texture){}
    CUDA T get(vec2 p) const {
        T res;
        tex2D<Type>(reinterpret_cast<Type*>(&res),mTexture,p.x,p.y);
        return res;
    }
    CUDA T getCubeMap(vec3 p) const {
        T res;
        texCubemap<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, p.z);
        return res;
    }
private:
    cudaTextureObject_t mTexture;
};

template<typename T>
class BuiltinSampler final:Uncopyable {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    cudaTextureObject_t mTexture;
public:
    BuiltinSampler(BuiltinArray<T>& array,
        cudaTextureAddressMode am = cudaAddressModeWrap,vec4 borderColor={},
        cudaTextureFilterMode fm = cudaFilterModeLinear,unsigned int maxAnisotropy = 0,
        bool sRGB = false):mArray(array.get()){
        cudaResourceDesc RD;
        RD.res.array.array = mArray;
        RD.resType = cudaResourceType::cudaResourceTypeArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.addressMode[0] = TD.addressMode[1] = am;
        *reinterpret_cast<vec4*>(TD.borderColor)= borderColor;
        TD.filterMode = fm;
        TD.maxAnisotropy = maxAnisotropy;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        TD.sRGB = sRGB;
        checkError(cudaCreateTextureObject(&mTexture,&RD,&TD,nullptr));
    }
    BuiltinSamplerGPU<T> toSampler() const {
        return mTexture;
    }
    ~BuiltinSampler() {
        checkError(cudaDestroyTextureObject(mTexture));
    }
};

template<typename T>
class BuiltinRenderTargetGPU final {
public:
    using Type = typename Rename<T>::Type;
    CUDA BuiltinRenderTargetGPU() = default;
    BuiltinRenderTargetGPU(cudaSurfaceObject_t target) :mTarget(target) {}
    CUDA T get(ivec2 p) const {
        auto res = surf2Dread<Type>(mTarget, p.x*sizeof(Type), p.y, cudaBoundaryModeClamp);
        return *reinterpret_cast<T*>(&res);
    }
    CUDA void set(ivec2 p,T v) {
        auto val = *reinterpret_cast<Type*>(&v);
        surf2Dwrite(val, mTarget, p.x*sizeof(Type), p.y, cudaBoundaryModeZero);
    }
private:
    cudaSurfaceObject_t mTarget;
};

namespace Impl {
    template<typename T>
    CALLABLE void clear(BuiltinRenderTargetGPU<T> rt,T val) {
        uvec2 p = { blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y};
        rt.set(p, val);
    }
}

template<typename T>
class BuiltinRenderTarget final:Uncopyable {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    cudaSurfaceObject_t mTarget;
    uvec2 mSize;
public:
    BuiltinRenderTarget(cudaArray_t array,uvec2 size) :mArray(array),mSize(size) {
        cudaResourceDesc RD;
        RD.res.array.array = mArray;
        RD.resType = cudaResourceType::cudaResourceTypeArray;
        checkError(cudaCreateSurfaceObject(&mTarget,&RD));
    }
    BuiltinRenderTarget(BuiltinArray<T>& array):BuiltinRenderTarget(array.get(),array.size()) {}
    BuiltinRenderTargetGPU<T> toTarget() const {
        return mTarget;
    }
    void clear(CommandBuffer& buffer, T val) {
        uint mul = sqrt(getEnvironment().getProp().maxThreadsPerBlock);
        dim3 grid(calcSize(mSize.x, mul), calcSize(mSize.y, mul));
        dim3 block(mul, mul);
        buffer.runKernelDim(Impl::clear<T>, grid, block, toTarget(), val);
    }
    void clear(Stream& stream,T val) {
        uint mul = sqrt(getEnvironment().getProp().maxThreadsPerBlock);
        dim3 grid(calcSize(mSize.x,mul),calcSize(mSize.y,mul));
        dim3 block(mul, mul);
        stream.runDim(Impl::clear<T>, grid, block, toTarget(), val);
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

template<typename T>
class BuiltinCubeMap final:Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaArray_t mArray;
    cudaTextureObject_t mTexture;
public:
    BuiltinCubeMap(size_t size) {
        auto desc = cudaCreateChannelDesc<Type>();
        cudaExtent extent{ size,size,6 };
        checkError(cudaMalloc3DArray(&mArray,desc,extent,cudaArrayCubemap));
        cudaResourceDesc RD;
        RD.resType = cudaResourceTypeArray;
        RD.res.array.array = mArray;
        cudaTextureDesc TD;
        TD.filterMode = cudaFilterModeLinear;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        checkError(cudaCreateTextureObject(&mTexture,&RD,&TD,nullptr));
    }
    cudaArray_t get() const {
        return mArray;
    }
    BuiltinSamplerGPU<T> toSampler() const {
        return mTexture;
    }
};
