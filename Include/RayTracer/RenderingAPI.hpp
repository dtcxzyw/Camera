#pragma once
#include <Math/Math.hpp>
#include <Spectrum/SpectrumConfig.hpp>
#include <Core/Memory.hpp>

class Integrator;
class SceneDesc;
class Transform;
class RayGeneratorWrapper;
class SampleWeightLUT;

MemorySpan<Spectrum> renderFrame(Integrator& integrator,
    const SceneDesc& scene,const Transform& cameraTransform,
    const RayGeneratorWrapper& rayGenerator, const SampleWeightLUT& weightLUT, uvec2 size);
