#pragma once
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <d3d11.h>
#undef max
#undef min
#undef near
#undef far
#include <Base/CompileEnd.hpp>
#include <Base/Builtin.hpp>
#include <Base/Environment.hpp>

class D3D11Image final :Uncopyable {
private:
    cudaGraphicsResource_t mRes;
    ID3D11Texture2D* mTexture;
    uvec2 mSize;
public:
    D3D11Image();
    ~D3D11Image();
    uvec2 size() const;
    void resize(uvec2 size);
    cudaArray_t bind(cudaStream_t stream);
    void unbind(cudaStream_t stream);
    ID3D11Texture2D* get() const;
};

class D3D11Window :Singletion {
protected:
    HWND mHwnd;
    IDXGISwapChain* mSwapChain;
    ID3D11Device* mDevice;
    ID3D11DeviceContext* mDeviceContext;
    ID3D11RenderTargetView* mRenderTargetView;
    bool mEnableVSync;
    WNDCLASSEX mWc;
    std::mutex mMutex;
    D3D11Window();
    void createRTV();
    void cleanRTV();
    void reset(uvec2 size);
    friend D3D11Window& getD3D11Window();
    friend LRESULT WINAPI wndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    friend class D3D11Image;
public:
    void show(bool isShow);
    void present(D3D11Image& image);
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
