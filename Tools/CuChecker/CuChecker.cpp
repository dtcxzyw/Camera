#include <iostream>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
using namespace std::experimental::filesystem;
using Map = std::map <std::string, uint64_t>;

//TODO:Include Analysis

uint64_t hash(const path& file) {
    std::ifstream in(file.string());
    uint64_t res = 0;
    while (in)res = res * 13131 + in.get();
    return res;
}

Map load(const path& cache) {
    Map res;
    std::ifstream in(cache);
    while (in) {
        std::string file;
        uint64_t hash;
        in >> file >> hash;
        res.emplace(file, hash);
    }
    return res;
}

Map scan(const path& srcPath,const path& cache) {
    Map res;
    for (auto&& file : recursive_directory_iterator(srcPath))
        if(file.status().type()==file_type::regular) {
            if (file.path().extension() == ".cu")
                res.emplace(file.path().filename().string(), hash(file.path()));
        }
    std::ofstream out(cache);
    for (auto&& info : res)
        out << info.first << " " << info.second << std::endl;
    return res;
}

constexpr auto line = "------------------------------------------------------";

void clear(const path& projectPath, const path& srcPath, const path& root,const path& cache) {
    std::cout << line << std::endl;
    const auto project = root / "Projects" / projectPath;
    const auto src = root / srcPath;
    const auto name = projectPath.filename().string();
    std::cout << "Checking Project " << name << std::endl;
    const auto cacheFile = cache / (name + ".cucache");
    const auto oldFiles = load(cacheFile);
    const auto newFiles = scan(src, cacheFile);
    std::set<std::string> toRemove;
    for (auto&& info : newFiles) {
        const auto it = oldFiles.find(info.first);
        if (it == oldFiles.cend()) {
            std::cout << "Found new file " << info.first << "." << std::endl;
            toRemove.emplace(info.first+".obj");
        }
        else if (info.second != it->second) {
            std::cout << "File " << info.first<<" is out of date." << std::endl;
            toRemove.emplace(info.first+".obj");
        }
    }
    if (toRemove.empty())std::cout << "Already up to date." << std::endl;
    else {
        for (auto&& file : recursive_directory_iterator(project))
            if (file.status().type() == file_type::regular) {
                const auto objectName = file.path().filename().string();
                if (toRemove.count(objectName)) {
                    std::cout << "Remove object file " << objectName << "." << std::endl;
                    remove(file.path());
                }
            }
    }
}

int main() {
    std::cout << "CuChecker (built " << __DATE__ << " at " << __TIME__ << ")" << std::endl;
    const auto root = path(__FILE__).parent_path().parent_path().parent_path();
    const auto cache = root / "Tools" / "Bin" / "CuCache";
    create_directory(cache);
    //Core
    clear("Core", "Src", root, cache);
    //Examples
    for (auto&& file : directory_iterator(root / "Examples"))
        if (file.status().type() == file_type::directory) {
            const auto projectName = "Examples" / file.path().filename();
            clear(projectName, projectName, root, cache);
        }

    std::cout << line << std::endl;
    std::cout << "Done." << std::endl << std::endl;
    std::cin.get();
    return 0;
}
