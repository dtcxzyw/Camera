#pragma once
#include <Base/Config.hpp>
#include <Base/Common.hpp>
#include <Base/DispatchSystem.hpp>
#include <vector>
#include <thread>

enum class GraphicsInteroperability {
    None
#ifdef CAMERA_D3D11_SUPPORT
    , D3D11
#endif
#ifdef CAMERA_OPENGL_SUPPORT
    , OpenGL
#endif
};

enum class AppType {
    Online, Offline
};

class Environment final :public Singletion<Environment> {
private:
    friend class Singletion<Environment>;
    Environment();
    std::vector<std::thread> mDevices;
    CommandBufferQueue mQueue;
    bool mRunning;
public:
    void init(AppType app = AppType::Offline,
        GraphicsInteroperability interop = GraphicsInteroperability::None);
    Future submit(std::unique_ptr<CommandBuffer> buffer);
    size_t queueSize() const;
    void uninit();
    ~Environment();
};
