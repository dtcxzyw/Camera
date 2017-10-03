#pragma once
#include "Common.hpp"
#define __CUDACC__
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_indirect_functions.h>
#include <cuda_surface_types.h>
#include <surface_indirect_functions.h>
#include "Pipeline.hpp"

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
class BuiltinSamplerGPU final {
public:
    using Type = typename Rename<T>::Type;
    BuiltinSamplerGPU() {}
    BuiltinSamplerGPU(cudaTextureObject_t texture):mTexture(texture){}
    CUDA T get(vec2 p) const {
        auto res=tex2D<Type>(mTexture,p.x,p.y);
        return *reinterpret_cast<T*>(&res);
    }
private:
    cudaTextureObject_t mTexture;
};

template<typename T>
class BuiltinSampler final {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    cudaTextureObject_t mTexture;
public:
    BuiltinSampler(size_t width, size_t height,const T* ptr,
        cudaTextureAddressMode am = cudaAddressModeWrap,vec4 borderColor={},
        cudaTextureFilterMode fm = cudaFilterModeLinear, int anisotropy = 0,
        bool norm = false, bool sRGB = false){
        auto desc=cudaCreateChannelDesc<Type>();
        checkError(cudaMallocArray(&mArray, desc, width, height));
        checkError(cudaMemcpy2DToArray(mArray, 0, 0, ptr, 0, width, height, cudaMemcpyHostToDevice));
        cudaResourceDesc RD;
        RD.res.array.array = mArray;
        RD.resType = cudaResourceType::cudaResourceTypeArray;
        cudaTextureDesc TD;
        TD.addressMode[0] = TD.addressMode[1] = TD.addressMode[2] = am;
        *reinterpret_cast<vec4*>(TD.borderColor)= borderColor;
        TD.filterMode = fm;
        TD.maxAnisotropy = anisotropy;
        TD.normalizedCoords = norm;
        TD.readMode = cudaReadModeElementType;
        TD.sRGB = sRGB;
        checkError(cudaCreateTextureObject(&mTexture,&RD,&TD,nullptr));
    }
    BuiltinSamplerGPU<T> toSampler() const {
        return mTexture;
    }
    ~BuiltinSampler() {
        checkError(cudaDestroyTextureObject(mTexture));
        checkError(cudaFreeArray(mArray));
    }
};

template<typename T>
class BuiltinRenderTargetGPU final {
public:
    using Type = typename Rename<T>::Type;
    BuiltinRenderTargetGPU(){}
    BuiltinRenderTargetGPU(cudaSurfaceObject_t target) :mTarget(target) {}
    CUDA T get(ivec2 p) const {
        auto res = surf2Dread<Type>(mTarget, p.x*sizeof(Type), p.y, cudaBoundaryModeZero);
        return *reinterpret_cast<T*>(&res);
    }
    CUDA void set(ivec2 p,T v) {
        auto val = *reinterpret_cast<Type*>(&v);
        surf2Dwrite(val, mTarget, p.x*sizeof(Type), p.y, cudaBoundaryModeClamp);
    }
private:
    cudaSurfaceObject_t mTarget;
};

namespace Impl {
    template<typename T>
    CALLABLE void clear(BuiltinRenderTargetGPU<T> rt,T val) {
        uvec2 p = { blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y };
        rt.set(p, val);
    }
}

template<typename T>
class BuiltinRenderTarget final {
private:
    cudaArray_t mArray;
    using Type = typename Rename<T>::Type;
    cudaSurfaceObject_t mTarget;
    uvec2 mSize;
public:
    BuiltinRenderTarget(size_t width, size_t height) :mSize(width,height) {
        auto desc = cudaCreateChannelDesc<Type>();
        checkError(cudaMallocArray(&mArray, &desc, width, height));
        cudaResourceDesc RD;
        RD.res.array.array = mArray;
        RD.resType = cudaResourceType::cudaResourceTypeArray;
        checkError(cudaCreateSurfaceObject(&mTarget,&RD));
    }
    BuiltinRenderTargetGPU<T> toTarget() const {
        return mTarget;
    }
    void clear(Pipeline& pipeline,T val) {
        uint mul = sqrt(getDevice().getProp().maxThreadsPerBlock);
        dim3 grid(calcSize(mSize.x,mul),calcSize(mSize.y,mul));
        dim3 block(mul, mul);
        pipeline.runDim(Impl::clear<T>, grid, block, toTarget(), val);
    }
    DataViewer<T> download(Pipeline& pipeline) const {
        auto res = allocBuffer<T>(mSize.x*mSize.y);
        checkError(cudaMemcpyFromArrayAsync(res.begin(),mArray,0,0,mSize.x*mSize.y*sizeof(T)
            ,cudaMemcpyDefault,pipeline.getId()));
        return res;
    }
    uvec2 size() const {
        return mSize;
    }
    cudaArray_const_t get() const {
        return mArray;
    }
    ~BuiltinRenderTarget() {
        checkError(cudaDestroySurfaceObject(mTarget));
        checkError(cudaFreeArray(mArray));
    }
};

