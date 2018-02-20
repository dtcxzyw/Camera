#pragma once
#include <Base/Common.hpp>
#include <vector>
#include <thread>
#include "ScanLineRenderer/DepthBuffer.hpp"

class Environment final :Singletion {
private:
    Environment();
    friend Environment& getEnvironment();
    std::vector<std::thread> mDevices;
    CommandBufferQueue mQueue;
    bool mRunning;
public:
    void init(size_t streamNum);
    Future submit(std::unique_ptr<CommandBuffer> buffer);
    size_t queueSize() const;
    void uninit();
    ~Environment();
};

Environment& getEnvironment();
