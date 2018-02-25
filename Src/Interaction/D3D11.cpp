#include <Base/CompileBegin.hpp>
#include <Interaction/D3D11.hpp>
#include <cuda_d3d11_interop.h>
#include <IMGUI/imgui.h>
#include <IMGUI/imgui_impl_dx11.h>
#include <Base/CompileEnd.hpp>
#include <stdexcept>

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

D3D11Window::D3D11Window() : mHwnd(nullptr), mSwapChain(nullptr),mDevice(nullptr), 
    mDeviceContext(nullptr), mRenderTargetView(nullptr), mStream(nullptr), mFrameBuffer(nullptr), 
    mRes(nullptr), mArray(nullptr), mEnableVSync(false), mWc({}) {
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

    constexpr auto flag = D3D11_CREATE_DEVICE_DEBUG;
    //constexpr auto flag = 0;

    D3D_FEATURE_LEVEL level;
    const D3D_FEATURE_LEVEL featureLevelArray[1] = {D3D_FEATURE_LEVEL_11_1};
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flag,
                                      featureLevelArray, 1, D3D11_SDK_VERSION, &sd, &mSwapChain, &mDevice,
                                      &level, &mDeviceContext) != S_OK)
        throw std::runtime_error("Failed to create D3D11 device.");
    
    mDevice->SetExceptionMode(D3D11_RAISE_FLAG_DRIVER_INTERNAL_ERROR);

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

    mRenderTargetView->GetResource(&mFrameBuffer);
}

void D3D11Window::cleanRTV() {
    if (mRenderTargetView) {
        if (mFrameBuffer) {
            if (mRes) {
                mArray = nullptr;
                if(mStream)checkError(cudaGraphicsUnmapResources(1, &mRes, mStream));
                checkError(cudaGraphicsUnregisterResource(mRes));
                mRes = nullptr;
            }
            mFrameBuffer->Release();
            mFrameBuffer = nullptr;
        }
        mRenderTargetView->Release();
        mRenderTargetView = nullptr;
    }
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

ID3D11Device* D3D11Window::getDevice() {
    return mDevice;
}

void D3D11Window::show(const bool isShow) {
    ShowWindow(mHwnd, isShow ? SW_SHOWDEFAULT : SW_HIDE);
    UpdateWindow(mHwnd);
}

void D3D11Window::bindBackBuffer(cudaStream_t stream) {
    if (mStream)throw std::logic_error("Can not bind buffers to two streams.");
    mStream = stream;
}

void D3D11Window::unbindBackBuffer() {
    if (mRes) {
        mArray = nullptr;
        checkError(cudaGraphicsUnmapResources(1, &mRes, mStream));
        checkError(cudaGraphicsUnregisterResource(mRes));
        mRes = nullptr;
    }
    mStream = nullptr;
}

void D3D11Window::present(cudaArray_t image) {
    cudaMemcpy3DParms parms = { 0 };
    parms.srcArray = image;
    parms.dstArray = getBackBuffer();
    parms.kind = cudaMemcpyDeviceToDevice;
    const auto fsiz = size();
    parms.extent = make_cudaExtent(fsiz.x, fsiz.y, 1);
    parms.srcPos = parms.dstPos = make_cudaPos(0, 0, 0);
    checkError(cudaMemcpy3DAsync(&parms, mStream));
    checkError(cudaStreamSynchronize(mStream));
}

cudaArray_t D3D11Window::getBackBuffer() {
    if(mRes==nullptr) {
            checkError(cudaGraphicsD3D11RegisterResource(&mRes,
                mFrameBuffer, cudaGraphicsRegisterFlagsNone));
        checkError(cudaGraphicsMapResources(1, &mRes, mStream));
    }
    if (mArray == nullptr) 
        checkError(cudaGraphicsSubResourceGetMappedArray(&mArray,mRes, 0, 0));
    return mArray;
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
