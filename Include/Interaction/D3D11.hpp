#pragma once
#include <Base/CompileBegin.hpp>
#include <d3d11.h>
#undef max
#undef min
#undef near
#undef far
#include <Base/CompileEnd.hpp>
#include <Base/Common.hpp>
#include <Base/Math.hpp>

class D3D11Window final:Singletion {
private:
    HWND mHwnd;
    IDXGISwapChain* mSwapChain;
    ID3D11Device* mDevice;
    ID3D11DeviceContext* mDeviceContext;
    ID3D11RenderTargetView* mRenderTargetView;

    cudaStream_t mStream;
    ID3D11Resource* mFrameBuffer;
    cudaGraphicsResource_t mRes;
    cudaArray_t mArray;

    bool mEnableVSync;
    WNDCLASSEX mWc;
    D3D11Window();
    void createRTV();
    void cleanRTV();
    cudaArray_t getBackBuffer();
    friend D3D11Window& getD3D11Window();
public:
    void reset(uvec2 size);
    ID3D11Device* getDevice();

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
    void renderGUI();
    ~D3D11Window();
};

D3D11Window& getD3D11Window();
