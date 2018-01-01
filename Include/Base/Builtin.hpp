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
    BuiltinArray(size_t width, size_t height,int flag=cudaArrayDefault) :mSize(width, height) {
        auto desc = cudaCreateChannelDesc<Type>();
        checkError(cudaMallocArray(&mArray, &desc, width, height,flag));
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
    uvec2 size() const {
        return mSize;
    }
    cudaArray_t get() const {
        return mArray;
    }
};

inline size_t calcMaxMipmapLevel(size_t width,size_t height) {
    return 1 + floor(std::log2(std::max(width,height)));
}

template<typename T>
class BuiltinMipmapedArray final :Uncopyable {
private:
    cudaMipmappedArray_t mArray;
    using Type = typename Rename<T>::Type;
    uvec2 mSize;
    void downSample(cudaArray_t src, cudaArray_t dst,uvec2 size,Stream& stream);
    void genMipmaps(const void* src, size_t level,Stream& stream) {
        cudaArray_t srcArray;
        {
            cudaMemcpy3DParms parms;
            parms.extent = { mSize.x,mSize.y,0 };
            parms.kind = cudaMemcpyHostToDevice;
            parms.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), mSize.x * sizeof(T), mSize.x, mSize.y);
            checkError(cudaGetMipmappedArrayLevel(&srcArray, mArray, 0));
            parms.dstArray = srcArray;
            checkError(cudaMemcpy3D(&parms));
        }
        uvec2 size = mSize;
        for (size_t i = 1; i <level; ++i) {
            size = max(size/2U,uvec2{ 1,1 });
            cudaArray_t dstArray;
            checkError(cudaGetMipmappedArrayLevel(&dstArray, mArray, i));
            downSample(srcArray, dstArray,size,stream);
            srcArray = dstArray;
        }
        stream.sync();
    }
public:
    BuiltinMipmapedArray(const void* src,size_t width,size_t height,Stream& stream,
        int flags=cudaArrayDefault,size_t level=0) :mSize(width, height) {
        auto desc = cudaCreateChannelDesc<Type>();
        auto maxLevel = calcMaxMipmapLevel(width, height);
        if (level == 0)level = maxLevel;
        else level = std::max(level, maxLevel);
        checkError(cudaMallocMipmappedArray(&mArray, &desc,
            make_cudaExtent(width, height,0),level,flags));
        genMipmaps(src, level,stream);
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

template<typename T>
class BuiltinSamplerGPU final {
public:
    using Type = typename Rename<T>::Type;
    CUDA BuiltinSamplerGPU() {};
    BuiltinSamplerGPU(cudaTextureObject_t texture):mTexture(texture){}
    CUDA T get(vec2 p) const {
        T res;
        tex2D<Type>(reinterpret_cast<Type*>(&res),mTexture,p.x,p.y);
        return res;
    }
    CUDA T getGather(vec2 p,int comp) const {
        T res;
        tex2Dgather<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y,comp);
        return res;
    }
    CUDA T getGrad(vec2 p, float dPdx,float dPdy) const {
        T res;
        tex2DGrad<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y, dPdx,dPdy);
        return res;
    }
    CUDA T getLod(vec2 p,float lod) const {
        T res;
        tex2DLod<Type>(reinterpret_cast<Type*>(&res), mTexture, p.x, p.y,lod);
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

//The trick comes from https://learnopengl.com/#!PBR/IBL/Diffuse-irradiance.
CUDAInline vec2 calcHDRUV(vec3 p) {
    return {atan(p.z,p.x)*0.1591f+0.5f,asin(p.y)*0.3183f+0.5f};
}

template<typename T>
class BuiltinSampler final:Uncopyable {
private:
    using Type = typename Rename<T>::Type;
    cudaTextureObject_t mTexture;
public:
    BuiltinSampler(cudaArray_t array,
        cudaTextureAddressMode am = cudaAddressModeWrap,vec4 borderColor={},
        cudaTextureFilterMode fm = cudaFilterModeLinear,unsigned int maxAnisotropy = 0,
        bool sRGB = false){
        cudaResourceDesc RD;
        RD.res.array.array = array;
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
    BuiltinSampler(BuiltinMipmapedArray<T>& array,
        cudaTextureAddressMode am = cudaAddressModeWrap, vec4 borderColor = {},
        cudaTextureFilterMode fm = cudaFilterModeLinear, unsigned int maxAnisotropy = 0,
        bool sRGB = false) {
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
        TD.maxMipmapLevelClamp=calcMaxMipmapLevel(size.x,size.y);
        TD.minMipmapLevelClamp=1.0f;
        TD.mipmapFilterMode=fm;
        checkError(cudaCreateTextureObject(&mTexture, &RD, &TD, nullptr));
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
    CUDA BuiltinRenderTargetGPU() {};
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
        checkError(cudaMalloc3DArray(&mArray,&desc,extent,cudaArrayCubemap));
        cudaResourceDesc RD;
        RD.resType = cudaResourceTypeArray;
        RD.res.array.array = mArray;
        cudaTextureDesc TD;
        memset(&TD, 0, sizeof(TD));
        TD.filterMode = cudaFilterModeLinear;
        TD.normalizedCoords = 1;
        TD.readMode = cudaReadModeElementType;
        checkError(cudaCreateTextureObject(&mTexture,&RD,&TD,nullptr));
    }
    ~BuiltinCubeMap() {
        checkError(cudaDestroyTextureObject(mTexture));
    }
    cudaArray_t get() const {
        return mArray;
    }
    BuiltinSamplerGPU<T> toSampler() const {
        return mTexture;
    }
};

namespace Impl {
    template<typename T>
    CALLABLE void downSample(BuiltinSamplerGPU<T> src,BuiltinRenderTargetGPU<T> rt) {
        uvec2 p = { blockIdx.x*blockDim.x + threadIdx.x,blockIdx.y*blockDim.y + threadIdx.y };
        uvec2 base = { p.x * 2,p.y * 2 };
        T val = {};
        for (auto i = 0; i < 2; ++i)
            for (auto j = 0; j < 2; ++j)
                val += src.get(base + uvec2{i, j});
        rt.set(p, val*0.25f);
    }
}

template<typename T>
inline void BuiltinMipmapedArray<T>::downSample(cudaArray_t src, cudaArray_t dst,
    uvec2 size,Stream& stream) {
    BuiltinSampler<T> sampler(src);
    BuiltinRenderTarget<T> RT(dst,size);
    uint mul = sqrt(getEnvironment().getProp().maxThreadsPerBlock);
    dim3 grid(calcSize(size.x, mul), calcSize(size.y, mul));
    dim3 block(mul, mul);
    stream.runDim(Impl::downSample<T>,grid,block,sampler.toSampler(),RT.toTarget());
}
