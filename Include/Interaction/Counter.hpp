#pragma once
#include <chrono>

class Counter final {
private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point mTime;
public:
    Counter();
    //return time elapsed, in seconds
    float record();
};

