#pragma once
#include <Core/DispatchSystem.hpp>
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
    std::vector<std::pair<size_t, size_t>> mMemInfo;
    CommandBufferQueue mQueue;
    std::unique_ptr<DispatchSystem> mMainDispatchSystem;
    std::thread::id mMainThread;
    bool mRunning;
    AppType mAppType;
public:
    void init(AppType app = AppType::Offline,
        GraphicsInteroperability interop = GraphicsInteroperability::None);
    AppType getAppType() const;
    Future submit(std::unique_ptr<CommandBuffer> buffer);
    size_t queueSize() const;
    std::pair<size_t,size_t> getMemInfo() const;
    void yield();
    bool isMainThread() const;
    void uninit();
    ~Environment();
};

class DeviceMonitor final :Uncopyable {
private:
    cudaDeviceProp mProp{};
    int mId;
    size_t mFree,mTotal;
    uintmax_t mTick;
    DeviceMonitor();
    void update();
    friend DeviceMonitor& getDeviceMonitor();
public:
    uintmax_t tick();
    int getId() const;
    const cudaDeviceProp& getProp() const;
    size_t getMemoryFreeSize() const;
    size_t getMemoryTotalSize() const;
};

DeviceMonitor& getDeviceMonitor();
