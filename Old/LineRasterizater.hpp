#pragma once
#include "ScanLine.hpp"

inline CUDA int ipart(float x) {
    return static_cast<int>(x);
}

inline CUDA float fpart(float x) {
    return abs(x - ipart(x));
}

inline CUDA float rfpart(float x) {
    return 1.0f - fpart(x);
}

template<typename Out, typename Uniform, typename FrameBuffer,
    FSF<Out, Uniform, FrameBuffer> fs>
    CUDA void drawLine(vec4 begin, vec4 end, const Out& bin, const Out& ein
        , const Uniform& uniform, FrameBuffer& frameBuffer) {
    auto pixel = frameBuffer.size();
    auto x1 = begin.x*pixel.x, y1 = begin.y*pixel.y, x2 = end.x*pixel.x, y2 = end.y*pixel.y;
    auto dx = x2 - x1;
    auto dy = y2 - y1;
    if (abs(dx) < abs(dy)) {
        swap(x1, y1);
        swap(x2, y2);
        swap(dx, dy);
    }
    if (x2 < x1) {
        swap(x1, x2);
        swap(y1, y2);
    }

    auto gradient = dy / dx;
    // handle first endpoint
    auto xend = round(x1);
    auto yend = y1 + gradient * (xend - x1);
    auto xgap = rfpart(x1 + 0.5);
    auto xpxl1 = xend;  // this will be used in the main loop
    auto ypxl1 = ipart(yend);
    vec2 b = { begin.x,begin.y }, e = { end.x,end.y };
    auto totl = distance(b, e);
    auto plot = [&](float x, float y, float w) {
        auto ew = distance(b, { x / pixel.x,y / pixel.y }) / totl;
        fs(begin*(1.0f - ew) + end*ew, bin*(1.0f - ew) + ein*ew, w, uniform, frameBuffer);
    };
    plot(xpxl1, ypxl1, rfpart(yend) * xgap);
    plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap);
    auto intery = yend + gradient; // first y-intersection for the main loop
                                   // handle second endpoint
    xend = round(x2);
    yend = y2 + gradient * (xend - x2);
    xgap = fpart(x2 + 0.5);
    auto xpxl2 = xend;  // this will be used in the main loop
    auto ypxl2 = ipart(yend);
    plot(xpxl2, ypxl2, rfpart(yend) * xgap);
    plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap);
    // main loop
    for (auto x = xpxl1 + 1; x <= xpxl2 - 1; ++x) {
        plot(x, ipart(intery), rfpart(intery));
        plot(x, ipart(intery) + 1, fpart(intery));
        intery += gradient;
    }
}

