#pragma once
#include <Base/CompileBegin.hpp>
#include <d3d11.h>
#undef max
#undef min
#undef near
#undef far
#include <Base/CompileEnd.hpp>
#include <Interaction/BoundImage.hpp>

class D3D11Image final :public BoundImage{
private:
    ID3D11Texture2D* mTexture;
    void reset() override;
public:
    D3D11Image();
    ~D3D11Image();
    cudaArray_t bind(cudaStream_t stream) override;
    void unbind(cudaStream_t stream) override;
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
    friend D3D11Window& getD3D11Window();
public:
    ID3D11Device * getDevice();
    std::mutex& getMutex();
    void reset(uvec2 size);

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
