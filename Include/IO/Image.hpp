#pragma once
#include <Base/Builtin.hpp>

std::shared_ptr<BuiltinArray<RGBA>> loadRGBA(const std::string& path,Stream& stream);
