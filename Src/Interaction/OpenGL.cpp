#include <Base/Config.hpp>
#ifdef CAMERA_OPENGL_SUPPORT
#include <Base/CompileBegin.hpp>
#include <GL/glew.h>
#include <Base/CompileEnd.hpp>
#include <Interaction/OpenGL.hpp>
#include <Base/CompileBegin.hpp>
#include <cuda_gl_interop.h>
#include <IMGUI/imgui.h>
#include <Base/CompileEnd.hpp>
#include <stdexcept>
#include <iostream>

static void errorCallBack(const int code, const char* str) {
    printf("Error:code = %d reason:%s\n",code,str);
    throw std::runtime_error(str);
}

static const char* getClipboardText(void* userData) {
    return glfwGetClipboardString(static_cast<GLFWwindow*>(userData));
}

static void setClipboardText(void* userData, const char* text) {
    glfwSetClipboardString(static_cast<GLFWwindow*>(userData), text);
}

static void mouseButtonCallback(GLFWwindow*, const int button, const int action, int) {
    if (action == GLFW_PRESS && button >= 0 && button < 3)
        GLWindow::get().mPressed[button] = true;
}

static void scrollCallback(GLFWwindow*, double, const double y) {
    GLWindow::get().mWheel += static_cast<float>(y); // Use fractional mouse wheel.
}

static void keyCallback(GLFWwindow*, const int key, int, const int action, int) {
    auto&& io = ImGui::GetIO();
    if (action == GLFW_PRESS)
        io.KeysDown[key] = true;
    if (action == GLFW_RELEASE)
        io.KeysDown[key] = false;

    io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
    io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
    io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
    io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
}

static void charCallback(GLFWwindow*, const unsigned int c) {
    auto&& io = ImGui::GetIO();
    if (c > 0 && c < 0x10000)
        io.AddInputCharacter(static_cast<unsigned short>(c));
}

void APIENTRY glDebugOutput(const GLenum source, const GLenum type, const GLuint id,
    const GLenum severity, GLsizei, const GLchar *message, const void *) {
    using namespace std;

    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    cerr << "---------------\n";
    cerr << "Debug message (" << id << "): " << message << '\n';

    switch (source) {
    case GL_DEBUG_SOURCE_API:             cerr << "Source: API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   cerr << "Source: Window System"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: cerr << "Source: Shader Compiler"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:     cerr << "Source: Third Party"; break;
    case GL_DEBUG_SOURCE_APPLICATION:     cerr << "Source: Application"; break;
    case GL_DEBUG_SOURCE_OTHER:           cerr << "Source: Other"; break;
        default: ;
    } cerr << '\n';

    switch (type) {
    case GL_DEBUG_TYPE_ERROR:               cerr << "Type: Error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: cerr << "Type: Deprecated Behaviour"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  cerr << "Type: Undefined Behaviour"; break;
    case GL_DEBUG_TYPE_PORTABILITY:         cerr << "Type: Portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE:         cerr << "Type: Performance"; break;
    case GL_DEBUG_TYPE_MARKER:              cerr << "Type: Marker"; break;
    case GL_DEBUG_TYPE_PUSH_GROUP:          cerr << "Type: Push Group"; break;
    case GL_DEBUG_TYPE_POP_GROUP:           cerr << "Type: Pop Group"; break;
    case GL_DEBUG_TYPE_OTHER:               cerr << "Type: Other"; break;
        default: ;
    } cerr << '\n';

    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:         cerr << "Severity: high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM:       cerr << "Severity: medium"; break;
    case GL_DEBUG_SEVERITY_LOW:          cerr << "Severity: low"; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION: cerr << "Severity: notification"; break;
        default: ;
    } cerr << "\n\n" << flush;

    if ((type <= GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR || severity <= GL_DEBUG_SEVERITY_MEDIUM)
        && severity != GL_DEBUG_SEVERITY_NOTIFICATION) {
        debugBreak();
        throw std::runtime_error("OpenGL error.");
    }
}

class GLContext final:public Singletion<GLContext> {
private:
    bool mFlag;
    friend class Singletion<GLContext>;
    GLContext():mFlag(false) {
        if (glfwInit()==GLFW_FALSE)
            throw std::runtime_error("Failed to initialize glfw.");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwSetErrorCallback(errorCallBack);
    }
public:
    void makeContext(GLFWwindow* window) {
        thread_local static GLFWwindow* current = nullptr;
        if (current != window) {
            glfwMakeContextCurrent(window);
            glDisable(GL_FRAMEBUFFER_SRGB);
            current = window;
            if (!mFlag) {
                glewExperimental = true;
                if (glewInit() != GLEW_NO_ERROR)
                    throw std::runtime_error("Failed to initialize glew.");
#ifdef CAMERA_OPENGL_ENABLE_DEBUG_OUTPUT
                glEnable(GL_DEBUG_OUTPUT);
                glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
                glDebugMessageCallback(glDebugOutput, nullptr);
#else
                glDisable(GL_DEBUG_OUTPUT);
                glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
                mFlag = true;
            }
        }
    }
    ~GLContext() {
        glfwTerminate();
    }
};

GLWindow::GLWindow() : mFBO(0), mWheel(0) {
    auto& context = GLContext::get();
    mWindow = glfwCreateWindow(800, 600, "OpenGL Viewer", nullptr, nullptr);
    if (!mWindow)
        throw std::runtime_error("Failed to create a window.");
    context.makeContext(mWindow);
    glfwSwapInterval(0);
    glGenFramebuffers(1, &mFBO);

    auto&& io = ImGui::GetIO();

    io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
    io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
    io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
    io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
    io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
    io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
    io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
    io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
    io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
    io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
    io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
    io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
    io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;

    io.ClipboardUserData = mWindow;

    io.SetClipboardTextFn = setClipboardText;
    io.GetClipboardTextFn = getClipboardText;

    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);
    glfwSetKeyCallback(mWindow, keyCallback);
    glfwSetCharCallback(mWindow, charCallback);
}

void GLWindow::makeContext() {
    GLContext::get().makeContext(mWindow);
}

/*
void GLWindow::present(GLImage& image) {
    makeContext();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D
        , image.get(), 0);
    const auto isiz = image.size();
    const auto frameSize = size();
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, frameSize.x, frameSize.y, 0, 0, isiz.x, isiz.y
        , GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}
*/

void GLWindow::setVSync(const bool enable) {
    makeContext();
    glfwSwapInterval(enable);
}

void GLWindow::swapBuffers() {
    glfwSwapBuffers(mWindow);
}

bool GLWindow::update() {
    makeContext();
    glfwPollEvents();
    return !glfwWindowShouldClose(mWindow);
}

void GLWindow::resize(const uvec2 size) {
    glfwSetWindowSize(mWindow, size.x,size.y);
}

uvec2 GLWindow::size() const {
    int w, h;
    glfwGetFramebufferSize(mWindow, &w, &h);
    return { w,h };
}

GLWindow::~GLWindow() {
    makeContext();
    ImGui::Shutdown();

    glDeleteFramebuffers(1, &mFBO);
    glfwDestroyWindow(mWindow);
}

void GLWindow::newFrame() {
    makeContext();
    auto&& io = ImGui::GetIO();

    int w, h,fw, fh;
    glfwGetWindowSize(mWindow, &w, &h);
    glfwGetFramebufferSize(mWindow, &fw, &fh);
    io.DisplaySize = { static_cast<float>(w),static_cast<float>(h) };
    io.DisplayFramebufferScale = { w > 0 ? (static_cast<float>(fw) / w) : 0,
            h > 0 ? (static_cast<float>(fh) / h) : 0 };

    io.DeltaTime = mCounter.record();

    if (glfwGetWindowAttrib(mWindow, GLFW_FOCUSED)) {
        if (io.WantMoveMouse) {
            glfwSetCursorPos(mWindow, io.MousePos.x, io.MousePos.y);
        }
        else {
            double mouseX, mouseY;
            glfwGetCursorPos(mWindow, &mouseX, &mouseY);
            io.MousePos = ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
        }
    }

    for (auto i = 0; i < 3; i++) {
        io.MouseDown[i] = mPressed[i] || glfwGetMouseButton(mWindow, i);
        mPressed[i] = false;
    }

    io.MouseWheel = mWheel;
    mWheel = 0.0f;

    // Hide OS mouse cursor if ImGui is drawing it
    glfwSetInputMode(mWindow, GLFW_CURSOR,
        io.MouseDrawCursor ? GLFW_CURSOR_HIDDEN : GLFW_CURSOR_NORMAL);

    // Start the frame. This call will update the io.WantCaptureMouse, io.WantCaptureKeyboard flag that you can use to dispatch inputs (or not) to your application.
    ImGui::NewFrame();
}
#endif
