#include <Camera/RealisticCamera.hpp>
#include <fstream>
#include <vector>
#include <sstream>

RealisticCamera::RealisticCamera(const std::string& lensDesc, const float apertureDiameter,
    const float focalDistance) :mApertureDiameter(apertureDiameter), mFocalDistance(focalDistance) {
    //read lens description file
    std::vector<LensElementInterface> lens;
    {
        std::ifstream data(lensDesc);
        std::string line;
        while (std::getline(data, line)) {
            if(!line.empty()) {
                if (line.front() == '#')continue;
                LensElementInterface cur;
                std::stringstream ss;
                ss << line;
                ss >> cur.curvatureRadius >> cur.thickness >> cur.eta >> cur.apertureRadius;
                lens.emplace_back(cur);
            }
        }
    }

}
