#pragma once
#include <ScanLineRenderer/ScanLine.hpp>

template<typename Vert, typename Uniform>
using GTSF = Vert(*)(Vert in, Uniform uniform);

template<typename Index,typename Vert,typename Uniform,GTSF<Vert,Uniform> gs>
auto calcTriangle(DataViewer<Vert> vert,Index idx,const Uniform* uniform,
    size_t inUnit,size_t outUnit) {
    assert(idx.size()%inUnit==0);
    auto nsiz = idx.size() / inUnit*outUnit*3;
    auto res = allocBuffer<Vert>(nsiz);

}
