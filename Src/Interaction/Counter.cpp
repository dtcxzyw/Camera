#include <Interaction/Counter.hpp>

Counter::Counter():mTime(Clock::now()) {}

float Counter::record() {
    const auto now = Clock::now();
    const auto delta = (now - mTime).count() * 1e-9f;
    mTime = now;
    return delta;
}
