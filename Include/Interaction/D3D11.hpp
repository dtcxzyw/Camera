#pragma once
#include <Core/Config.hpp>
#ifdef CAMERA_D3D11_SUPPORT
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <Interaction/Counter.hpp>

class HWND__;
using HWND = HWND__ * ;
class HINSTANCE__;
using HINSTANCE = HINSTANCE__ * ;

class IDXGISwapChain;
class ID3D11Device;
class ID3D11DeviceContext;
class ID3D11RenderTargetView;
class ID3D11Resource;

class D3D11Window final:public Singletion<D3D11Window> {
private:
    HWND mHwnd;
    HINSTANCE mInstance;
    IDXGISwapChain* mSwapChain;
    ID3D11Device* mDevice;
    ID3D11DeviceContext* mDeviceContext;
    ID3D11RenderTargetView* mRenderTargetView;

    cudaStream_t mStream;
    ID3D11Resource* mFrameBuffer;
    cudaGraphicsResource_t mRes;
    cudaArray_t mArray;

    bool mEnableVSync;

    Counter mCounter;

    friend class Singletion<D3D11Window>;

    D3D11Window();
    void createRTV();
    void cleanRTV();
    cudaArray_t getBackBuffer();
public:
    void reset(uvec2 fsiz);
    void enumDevices(int* buf,unsigned int* count) const;

    void bindBackBuffer(cudaStream_t stream);
    void unbindBackBuffer();
    void present(cudaArray_t image);

    void show(bool isShow);
    void setVSync(bool enable);
    void swapBuffers();
    bool update();
    void resize(uvec2 size);
    uvec2 size() const;
    void newFrame();
    ~D3D11Window();
};
#endif
