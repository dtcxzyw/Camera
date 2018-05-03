#pragma once
#include <Core/Builtin.hpp>
#include <Math/Interaction.hpp>

template<typename Type>
Type texture(const BuiltinSamplerRef<Type>& sampler, const Interaction& interaction) {
    return sampler.getGrad(interaction.uv,{},{});
}
