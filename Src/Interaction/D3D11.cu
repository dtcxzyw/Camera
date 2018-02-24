#include <Base/CompileBegin.hpp>
#include <Interaction/D3D11.hpp>
#include <cuda_d3d11_interop.h>
#include <IMGUI/imgui.h>
#include <IMGUI/imgui_impl_dx11.h>
#include <Base/CompileEnd.hpp>

extern IMGUI_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI wndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            getD3D11Window().reset({LOWORD(lParam),HIWORD(lParam)});
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default: ;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

static void checkResult(const HRESULT res) {
    if (res != S_OK)__debugbreak();
}

D3D11Window::D3D11Window(): mRenderTargetView(nullptr), mEnableVSync(false) {
    constexpr auto title = L"D3D11 Viewer";
    mWc = {
        sizeof(WNDCLASSEX), CS_CLASSDC, wndProc, 0L, 0L,
        GetModuleHandle(nullptr), nullptr, LoadCursor(nullptr, IDC_ARROW), nullptr, nullptr,
        title, nullptr
    };
    RegisterClassEx(&mWc);
    mHwnd = CreateWindow(title,title, WS_OVERLAPPEDWINDOW, 100, 100, 800, 600,
        NULL, NULL, mWc.hInstance, NULL);

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = mHwnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = true;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    //constexpr auto flag = D3D11_CREATE_DEVICE_DEBUG;
    constexpr auto flag = 0;

    D3D_FEATURE_LEVEL level;
    const D3D_FEATURE_LEVEL featureLevelArray[1] = {D3D_FEATURE_LEVEL_11_1};
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flag,
                                      featureLevelArray, 1, D3D11_SDK_VERSION, &sd, &mSwapChain, &mDevice,
                                      &level, &mDeviceContext) != S_OK)
        throw std::runtime_error("Failed to create D3D11 device.");

    createRTV();

    if (!ImGui_ImplDX11_Init(mHwnd, mDevice, mDeviceContext))
        throw std::runtime_error("Failed to setup ImGui binding.");
}

void D3D11Window::createRTV() {
    DXGI_SWAP_CHAIN_DESC sd;
    checkResult(mSwapChain->GetDesc(&sd));

    ID3D11Texture2D* texture;
    checkResult(mSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                      reinterpret_cast<void**>(&texture)));
    D3D11_RENDER_TARGET_VIEW_DESC RTVD;
    ZeroMemory(&RTVD, sizeof(RTVD));
    RTVD.Format = sd.BufferDesc.Format;
    RTVD.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    checkResult(mDevice->CreateRenderTargetView(texture, &RTVD, &mRenderTargetView));
    texture->Release();
}

void D3D11Window::cleanRTV() {
    if (mRenderTargetView) {
        mRenderTargetView->Release();
        mRenderTargetView = nullptr;
    }
}

ID3D11Device* D3D11Window::getDevice() {
    return mDevice;
}

std::mutex& D3D11Window::getMutex() {
    return mMutex;
}

void D3D11Window::reset(const uvec2 nsiz) {
    if (size() != nsiz) {
        ImGui_ImplDX11_InvalidateDeviceObjects();
        cleanRTV();
        checkResult(mSwapChain->ResizeBuffers(0, nsiz.x, nsiz.y, DXGI_FORMAT_UNKNOWN, 0));
        createRTV();
        ImGui_ImplDX11_CreateDeviceObjects();
    }
}

void D3D11Window::show(const bool isShow) {
    ShowWindow(mHwnd, isShow ? SW_SHOWDEFAULT : SW_HIDE);
    UpdateWindow(mHwnd);
}

void D3D11Window::present(D3D11Image& image) {
    if (image.size() == size()) {
        std::lock_guard<std::mutex> guard(mMutex);
        ID3D11Resource* res;
        mRenderTargetView->GetResource(&res);
        mDeviceContext->CopyResource(res, image.get());
        res->Release();
    }
}

void D3D11Window::setVSync(const bool enable) {
    mEnableVSync = enable;
}

void D3D11Window::swapBuffers() {
    mSwapChain->Present(mEnableVSync, 0);
}

bool D3D11Window::update() {
    MSG msg;
    if (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return msg.message != WM_QUIT;
}

void D3D11Window::resize(const uvec2 size) {
    DXGI_SWAP_CHAIN_DESC desc;
    checkResult(mSwapChain->GetDesc(&desc));
    desc.BufferDesc.Width = size.x;
    desc.BufferDesc.Height = size.y;
    checkResult(mSwapChain->ResizeTarget(&desc.BufferDesc));
    reset(size);
}

uvec2 D3D11Window::size() const {
    DXGI_SWAP_CHAIN_DESC sd;
    checkResult(mSwapChain->GetDesc(&sd));
    return {sd.BufferDesc.Width, sd.BufferDesc.Height};
}

void D3D11Window::newFrame() {
    ImGui_ImplDX11_NewFrame();
}

void D3D11Window::renderGUI() {
    std::lock_guard<std::mutex> guard(mMutex);
    mDeviceContext->OMSetRenderTargets(1, &mRenderTargetView, nullptr);
    ImGui::Render();
}

D3D11Window::~D3D11Window() {
    ImGui_ImplDX11_Shutdown();
    cleanRTV();
    mSwapChain->Release();
    mDeviceContext->Release();
    mDevice->Release();
    UnregisterClass(L"D3D11 Viewer", mWc.hInstance);
}

D3D11Window& getD3D11Window() {
    static D3D11Window window;
    return window;
}

D3D11Image::D3D11Image(): mTexture(nullptr) {}

D3D11Image::~D3D11Image() {
    destoryRes();
    if (mTexture)mTexture->Release();
}

void D3D11Image::reset() {
    if (mTexture) {
        mTexture->Release();
        mTexture = nullptr;
    }
    auto&& window = getD3D11Window();
    D3D11_TEXTURE2D_DESC td;
    td.Width = mSize.x;
    td.Height = mSize.y;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.SampleDesc.Quality = 0;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET;
    td.CPUAccessFlags = 0;
    td.MiscFlags = 0;
    checkResult(window.getDevice()->CreateTexture2D(&td, nullptr, &mTexture));
    checkError(cudaGraphicsD3D11RegisterResource(&mRes, mTexture,
                                                    cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

cudaArray_t D3D11Image::bind(const cudaStream_t stream) {
    std::lock_guard<std::mutex> guard(getD3D11Window().getMutex());
    return BoundImage::bind(stream);
}

void D3D11Image::unbind(const cudaStream_t stream) {
    std::lock_guard<std::mutex> guard(getD3D11Window().getMutex());
    BoundImage::unbind(stream);
}

ID3D11Texture2D* D3D11Image::get() const {
    return mTexture;
}
