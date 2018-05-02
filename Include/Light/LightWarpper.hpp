#pragma once
#include <Light/LightingSample.hpp>
#include <Core/DeviceMemory.hpp>

struct LightWrapperLi {
    CUDA virtual ~LightWrapperLi() = default;
    CUDA virtual LightingSample sampleLi(vec2 sample,Point pos) const = 0;
};

struct LightWrapperLe {
    CUDA virtual ~LightWrapperLe() = default;
    CUDA virtual LightingSample sampleLe(const Ray& ray) const = 0;
};

namespace Impl {
    template<typename Light>
    struct LightWrapperLiImpl final :LightWrapperLi {
    private:
        Light mLight;
    public:
        template<typename... Args>
        CUDA explicit LightWrapperLiImpl(Args... args) :mLight(args...) {}
        CUDA LightingSample sampleLi(const vec2 sample, const Point pos) const override {
            return mLight.sampleLi(sample, pos);
        }
    };

    template<typename Light>
    struct LightWrapperLeImpl final :LightWrapperLe {
    private:
        Light mLight;
    public:
        template<typename... Args>
        CUDA explicit LightWrapperLeImpl(Args... args) :mLight(args...) {}
        CUDA LightingSample sampleLe(const Ray& ray) const override {
            return mLight.sampleLe(ray);
        }
    };
}

template<typename Light,typename... Args>
MemorySpan<LightWrapperLi> makeLightWrapperLi(CommandBuffer& buffer,Args&&... args) {
    auto res = constructOnDevice<Impl::LightWrapperLiImpl<Light>>(buffer, std::forward<Args>(args)...);
    return *reinterpret_cast<MemorySpan<LightWrapperLi>*>(&res);
}

template<typename Light, typename... Args>
MemorySpan<LightWrapperLe> makeLightWrapperLe(CommandBuffer& buffer, Args&&... args) {
    auto res = constructOnDevice<Impl::LightWrapperLeImpl<Light>>(buffer, std::forward<Args>(args)...);
    return *reinterpret_cast<MemorySpan<LightWrapperLe>*>(&res);
}

