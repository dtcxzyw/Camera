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
#include <Interaction/Counter.hpp>

class D3D11Window final:public Singletion<D3D11Window> {
private:
    HWND mHwnd;
    IDXGISwapChain* mSwapChain;
    ID3D11Device* mDevice;
    ID3D11DeviceContext* mDeviceContext;
    ID3D11RenderTargetView* mRenderTargetView;
    WNDCLASSEX mWc;

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
    ~D3D11Window();
};
