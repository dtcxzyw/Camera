#include <Base/Environment.hpp>
#include <Base/DispatchSystem.hpp>
#include <stdexcept>
#ifdef CAMERA_D3D11_SUPPORT
#include <Interaction/D3D11.hpp>
#endif
#ifdef CAMERA_OPENGL_SUPPORT
#include <Base/CompileBegin.hpp>
#include <cuda_gl_interop.h>
#include <cuda_d3d11_interop.h>
#include <Base/CompileEnd.hpp>
#endif

Environment::Environment() : mRunning(true) {}

static void setDevice(const int id) {
    checkError(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    checkError(cudaSetDeviceFlags(cudaDeviceLmemResizeToMax));
    checkError(cudaSetDevice(id));
}

static void resetDevice() {
    checkError(cudaDeviceSynchronize());
}

void Environment::init(const GraphicsInteroperability interop) {
    std::vector<int> devices;
    if(interop==GraphicsInteroperability::None) {
        int cnt;
        checkError(cudaGetDeviceCount(&cnt));
        for (auto id = 0; id < cnt; ++id)
            devices.emplace_back(id);
    }
    else {
        unsigned int deviceCount;
        int device[256];
#ifdef CAMERA_D3D11_SUPPORT
        if (interop == GraphicsInteroperability::D3D11)
            checkError(cudaD3D11GetDevices(&deviceCount, device, 256,
                D3D11Window::get().getDevice(), cudaD3D11DeviceListAll));
#endif
#ifdef CAMERA_OPENGL_SUPPORT
        if(interop==GraphicsInteroperability::OpenGL)
            checkError(cudaGLGetDevices(&deviceCount, device, 256, cudaGLDeviceListAll));
#endif
        devices.assign(device, device + deviceCount);
    }
    int firstDevice = -1;
    for (auto id : devices) {
        cudaDeviceProp prop{};
        checkError(cudaGetDeviceProperties(&prop, id));
        const auto ver = prop.major * 100 + prop.minor;
        if (ver < 305)continue;
        if (prop.computeMode != cudaComputeModeDefault)continue;
        firstDevice = id;
        mDevices.emplace_back([this,id]() {
            setDevice(id);
            {
                DispatchSystem system( mQueue);
                while (mRunning) system.update();
            }
            resetDevice();
        });
    }
    if (mDevices.empty())
        throw std::runtime_error("Failed to initialize the CUDA environment.");
    setDevice(firstDevice);
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
