#include <Interaction/D3D11.hpp>
#include <Base/Config.hpp>
#include <Base/CompileBegin.hpp>
#include <cuda_d3d11_interop.h>
#include <IMGUI/imgui.h>
#include <Base/CompileEnd.hpp>
#include <stdexcept>

static LRESULT setEvent(HWND hwnd, const UINT msg, const WPARAM wParam, 
    const LPARAM lParam) {
    auto&& io = ImGui::GetIO();

    const auto isAnyMouseButtonDown=[&io]{
        for (auto&& x : io.MouseDown)
            if (x)return true;
        return false;
    };

    switch (msg) {
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
    {
        auto button = 0;
        if (msg == WM_LBUTTONDOWN) button = 0;
        if (msg == WM_RBUTTONDOWN) button = 1;
        if (msg == WM_MBUTTONDOWN) button = 2;
        if (!isAnyMouseButtonDown() && GetCapture() == nullptr)SetCapture(hwnd);
        io.MouseDown[button] = true;
        return 0;
    }
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    {
        auto button = 0;
        if (msg == WM_LBUTTONUP) button = 0;
        if (msg == WM_RBUTTONUP) button = 1;
        if (msg == WM_MBUTTONUP) button = 2;
        io.MouseDown[button] = false;
        if (!isAnyMouseButtonDown() && GetCapture() == hwnd)
            ReleaseCapture();
        break;
    }
    case WM_MOUSEWHEEL:
        io.MouseWheel += GET_WHEEL_DELTA_WPARAM(wParam) > 0 ? +1.0f : -1.0f;
        break;
    case WM_MOUSEMOVE:
        io.MousePos.x = static_cast<signed short>(lParam);
        io.MousePos.y = static_cast<signed short>(lParam >> 16);
        break;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        if (wParam < 256)
            io.KeysDown[wParam] = true;
        break;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (wParam < 256)
            io.KeysDown[wParam] = false;
        break;
    case WM_CHAR:
        if (wParam > 0 && wParam < 0x10000)
            io.AddInputCharacter(static_cast<unsigned short>(wParam));
        break;
        default: break;
    }
    return 0;
}

static LRESULT WINAPI wndProc(HWND hWnd, const UINT msg, const WPARAM wParam, 
    const LPARAM lParam) {
    if (setEvent(hWnd, msg, wParam, lParam))return true;

    switch (msg) {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            D3D11Window::get().reset({LOWORD(lParam),HIWORD(lParam)});
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
    mDeviceContext(nullptr), mRenderTargetView(nullptr), mWc({}), mStream(nullptr), 
    mFrameBuffer(nullptr), mRes(nullptr), mArray(nullptr), mEnableVSync(false) {
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

#ifdef CAMERA_D3D11_ENABLE_DEBUG_LAYER
    constexpr auto flag = D3D11_CREATE_DEVICE_DEBUG;
#else
    constexpr auto flag = 0;
#endif

    D3D_FEATURE_LEVEL level;
    const D3D_FEATURE_LEVEL featureLevelArray[1] = {D3D_FEATURE_LEVEL_11_1};
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flag,
                                      featureLevelArray, 1, D3D11_SDK_VERSION, &sd, &mSwapChain, &mDevice,
                                      &level, &mDeviceContext) != S_OK)
        throw std::runtime_error("Failed to create D3D11 device.");
    
    mDevice->SetExceptionMode(D3D11_RAISE_FLAG_DRIVER_INTERNAL_ERROR);

    createRTV();

    //keyboard mapping
    auto&& io = ImGui::GetIO();
    io.KeyMap[ImGuiKey_Tab] = VK_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = VK_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = VK_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = VK_UP;
    io.KeyMap[ImGuiKey_DownArrow] = VK_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = VK_PRIOR;
    io.KeyMap[ImGuiKey_PageDown] = VK_NEXT;
    io.KeyMap[ImGuiKey_Home] = VK_HOME;
    io.KeyMap[ImGuiKey_End] = VK_END;
    io.KeyMap[ImGuiKey_Delete] = VK_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = VK_BACK;
    io.KeyMap[ImGuiKey_Enter] = VK_RETURN;
    io.KeyMap[ImGuiKey_Escape] = VK_ESCAPE;
    io.KeyMap[ImGuiKey_A] = 'A';
    io.KeyMap[ImGuiKey_C] = 'C';
    io.KeyMap[ImGuiKey_V] = 'V';
    io.KeyMap[ImGuiKey_X] = 'X';
    io.KeyMap[ImGuiKey_Y] = 'Y';
    io.KeyMap[ImGuiKey_Z] = 'Z';

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

void D3D11Window::reset(const uvec2 fsiz) {
    if (size() != fsiz) {
        cleanRTV();
        checkResult(mSwapChain->ResizeBuffers(0, fsiz.x, fsiz.y, DXGI_FORMAT_UNKNOWN, 0));
        createRTV();
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
    auto&& io = ImGui::GetIO();

    const auto fsiz = size();
    io.DisplaySize = {static_cast<float>(fsiz.x), static_cast<float>(fsiz.y)};

    io.DeltaTime = mCounter.record();

    // Read keyboard modifiers inputs
    io.KeyCtrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
    io.KeyShift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
    io.KeyAlt = (GetKeyState(VK_MENU) & 0x8000) != 0;
    io.KeySuper = false;

    if (io.WantMoveMouse) {
        POINT pos = {static_cast<LONG>(io.MousePos.x),static_cast<LONG>(io.MousePos.y) };
        ClientToScreen(mHwnd, &pos);
        SetCursorPos(pos.x, pos.y);
    }

    if (io.MouseDrawCursor)SetCursor(nullptr);

    ImGui::NewFrame();
}

D3D11Window::~D3D11Window() {
    ImGui::Shutdown();
    cleanRTV();
    mSwapChain->Release();
    mDeviceContext->Release();
    mDevice->Release();
    UnregisterClass(L"D3D11 Viewer", mWc.hInstance);
}
