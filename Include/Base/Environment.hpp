#pragma once
#include <Base/Common.hpp>
#include <Base/DispatchSystem.hpp>
#include <vector>
#include <thread>

enum class GraphicsInteroperability {
    None, D3D11, OpenGL
};

class Environment final :Singletion {
private:
    Environment();
    friend Environment& getEnvironment();
    std::vector<std::thread> mDevices;
    CommandBufferQueue mQueue;
    bool mRunning;
public:
    void init(GraphicsInteroperability interop=GraphicsInteroperability::None);
    Future submit(std::unique_ptr<CommandBuffer> buffer);
    size_t queueSize() const;
    void uninit();
    ~Environment();
};

Environment& getEnvironment();
