#include <Base/Environment.hpp>
#include <Base/DispatchSystem.hpp>
#include <stdexcept>
#include "Scene/ECS.hpp"
#ifdef CAMERA_D3D11_SUPPORT
#include <Interaction/D3D11.hpp>
#endif
#ifdef CAMERA_OPENGL_SUPPORT
#include <Base/CompileBegin.hpp>
#include <cuda_gl_interop.h>
#include <cuda_d3d11_interop.h>
#include <Base/CompileEnd.hpp>
#endif

static void setDevice(const int id, const AppType app) {
    const auto schedule=(app==AppType::Online? cudaDeviceScheduleSpin: 
        cudaDeviceScheduleBlockingSync);
    checkError(cudaSetDeviceFlags(schedule));
    checkError(cudaSetDevice(id));
}

static void resetDevice() {
    checkError(cudaDeviceSynchronize());
    clearMemoryPool();
    checkError(cudaDeviceReset());
}

Environment::Environment() :mRunning(false) {}

void Environment::init(const AppType app,const GraphicsInteroperability interop) {
    std::vector<int> devices;
    if (interop == GraphicsInteroperability::None) {
        int cnt;
        checkError(cudaGetDeviceCount(&cnt));
        for (auto id = 0; id < cnt; ++id)
            devices.emplace_back(id);
    }
    else {
        auto deviceCount = 0U;
        int device[256];
        #ifdef CAMERA_D3D11_SUPPORT
        if (interop == GraphicsInteroperability::D3D11)
            D3D11Window::get().enumDevices(device, &deviceCount);
        #endif
        #ifdef CAMERA_OPENGL_SUPPORT
        if (interop == GraphicsInteroperability::OpenGL)
            checkError(cudaGLGetDevices(&deviceCount, device, 256, cudaGLDeviceListAll));
        #endif
        devices.assign(device, device + deviceCount);
    }

    std::vector<int> choosed;
    for (auto id : devices) {
        cudaDeviceProp prop{};
        checkError(cudaGetDeviceProperties(&prop, id));
        const auto ver = prop.major * 100 + prop.minor;
        if (ver < 305)continue;
        if (prop.computeMode != cudaComputeModeDefault)continue;
        choosed.emplace_back(id);
        #ifdef CAMERA_SINGLE_STREAM
        break;
        #endif
    }

    size_t initCount = 0;
    auto idx = 0;
    for (auto id : choosed) {
        mDevices.emplace_back([this, id,idx, choosed, app, &initCount]() {
            setDevice(id, app);

            const auto isMainDevice = choosed.front() == id;
            for (auto&& dev : choosed)
                if (dev != id)
                    checkError(cudaDeviceEnablePeerAccess(dev, 0));

            ++initCount;
            while (!mRunning)std::this_thread::yield();

            {
                auto&& monitor = getDeviceMonitor();
                DispatchSystem system(mQueue, app == AppType::Offline);
                while (mRunning){ 
                    system.update();
                    monitor.tick();
                    mMemInfo[idx] = { monitor.getMemoryFreeSize(),monitor.getMemoryTotalSize() };
                }
                checkError(cudaDeviceSynchronize());
            }

            if (!isMainDevice)resetDevice();
        });

        ++idx;
    }
    if (mDevices.empty())
        throw std::runtime_error("Failed to initialize the CUDA environment.");
    setDevice(choosed.front(),app);
    while (initCount != choosed.size()) std::this_thread::yield();
    mMemInfo.resize(mDevices.size());
    mRunning = true;
}

Future Environment::submit(std::unique_ptr<CommandBuffer> buffer) {
    const auto promise = std::make_shared<Impl::TaskState>();
    mQueue.submit(promise, std::move(buffer));
    return Future{promise};
}

size_t Environment::queueSize() const {
    return mQueue.size();
}

std::pair<size_t, size_t> Environment::getMemInfo() const {
    size_t free=0,total=0;
    for (auto&& info : mMemInfo)
        free += info.first, total += info.second;
    return { free,total };
}

void Environment::uninit() {
    mRunning = false;
    for (auto&& device : mDevices)
        device.join();
    mDevices.clear();
    mQueue.clear();
}

Environment::~Environment() {
    resetDevice();
}

DeviceMonitor::DeviceMonitor(): mId(0), mFree(0), mTotal(1), mTick(0) {
    checkError(cudaGetDevice(&mId));
    checkError(cudaGetDeviceProperties(&mProp, mId));
}

void DeviceMonitor::update() {
    checkError(cudaMemGetInfo(&mFree,&mTotal));
}

uintmax_t DeviceMonitor::tick() {
    if ((mTick & 0xff) == 0)update();
    return ++mTick;
}

int DeviceMonitor::getId() const {
    return mId;
}

const cudaDeviceProp& DeviceMonitor::getProp() const {
    return mProp;
}

size_t DeviceMonitor::getMemoryFreeSize() const {
    return mFree;
}

size_t DeviceMonitor::getMemoryTotalSize() const {
    return mTotal;
}

DeviceMonitor& getDeviceMonitor() {
    thread_local static DeviceMonitor monitor;
    return monitor;
}
