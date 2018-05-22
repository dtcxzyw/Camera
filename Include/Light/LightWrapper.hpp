#pragma once
#include <Light/LightingSample.hpp>
#include <Core/DeviceMemory.hpp>

struct LightWrapper {
    DEVICE virtual ~LightWrapper() = default;
    DEVICE virtual LightingSample sampleLi(vec2 sample,Point pos) const = 0;
    DEVICE virtual Spectrum le(const Ray& ray) const = 0;
};

namespace Impl {
    template<typename Light>
    struct LightWrapperImpl final :LightWrapper {
    private:
        Light mLight;
    public:
        template<typename... Args>
        DEVICE explicit LightWrapperImpl(Args... args) :mLight(args...) {}
        DEVICE LightingSample sampleLi(const vec2 sample, const Point pos) const override {
            return mLight.sampleLi(sample, pos);
        }
        DEVICE Spectrum le(const Ray& ray) const override {
            return mLight.le(ray);
        }
    };
}

template<typename Light,typename... Args>
MemorySpan<LightWrapper> makeLightWrapper(Stream& stream,Args&&... args) {
    auto res = constructOnDevice<Impl::LightWrapperImpl<Light>>(stream, std::forward<Args>(args)...);
    return *reinterpret_cast<MemorySpan<LightWrapper>*>(&res);
}

