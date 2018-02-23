#include <Base/Environment.hpp>
#include <Base/DispatchSystem.hpp>
#include <stdexcept>

Environment::Environment() : mRunning(true) {}

static void setDevice(const int id) {
    checkError(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    checkError(cudaSetDeviceFlags(cudaDeviceLmemResizeToMax));
    checkError(cudaSetDevice(id));
}

static void resetDevice() {
    checkError(cudaDeviceSynchronize());
}

void Environment::init(size_t streamNum) {
    int cnt;
    checkError(cudaGetDeviceCount(&cnt));
    cudaDeviceProp prop{};
    for (auto id = 0; id < cnt; ++id) {
        checkError(cudaGetDeviceProperties(&prop, id));
        const auto ver = prop.major * 100 + prop.minor;
        if (ver < 305)continue;
        mDevices.emplace_back([this,streamNum,id]() {
            setDevice(id);
            {
                DispatchSystem system(streamNum, mQueue);
                while (mRunning) system.update();
            }
            resetDevice();
        });
    }
    if (mDevices.empty())
        throw std::runtime_error("Failed to initialize the CUDA environment.");
    //TODO:Choose a device which supports OpenGL/Direct3D.
    setDevice(0);
}

Future Environment::submit(std::unique_ptr<CommandBuffer> buffer) {
    const auto promise = std::make_shared<Impl::TaskState>();
    mQueue.submit(promise,std::move(buffer));
    return Future{promise};
}

size_t Environment::queueSize() const {
    return mQueue.size();
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

Environment& getEnvironment() {
    static Environment env;
    return env;
}
