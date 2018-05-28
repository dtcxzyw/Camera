#pragma once
#include <vector>

std::vector<char> loadLZ4(const std::string& path);
void saveLZ4(const std::string& path, const std::vector<char>& data);
