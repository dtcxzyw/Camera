#include <Base/CompileBegin.hpp>
#include <GL/glew.h>
#include <Interaction/OpenGL.hpp>
#include <cuda_gl_interop.h>
#include <IMGUI/imgui_impl_glfw_gl3.h>
#include <Base/CompileEnd.hpp>

namespace Impl {
    static void errorCallBack(const int code, const char* str) {
        printf("Error:code = %d reason:%s\n",code,str);
        throw std::runtime_error(str);
    }
}

class GLContext final:Singletion {
private:
    bool mFlag;
    GLContext():mFlag(false) {
        if (glfwInit()==GLFW_FALSE)
            throw std::runtime_error("Failed to initialize glfw.");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwSetErrorCallback(Impl::errorCallBack);
    }
    friend GLContext& getContext();
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
                mFlag = true;
            }
        }
    }
    ~GLContext() {
        glfwTerminate();
    }
};

static GLContext& getContext() {
    static GLContext context;
    return context;
}

GLWindow::GLWindow(GLWindow* share) {
    auto& context=getContext();
    mWindow = glfwCreateWindow(800, 600, "OpenGL Viewer", nullptr,
        share?share->mWindow:nullptr);
    if (!mWindow)
        throw std::runtime_error("Failed to create a window.");
    context.makeContext(mWindow);
    glfwSwapInterval(0);
    glGenFramebuffers(1, &mFBO);
}

void GLWindow::makeContext() {
    getContext().makeContext(mWindow);
}

void GLWindow::unmakeContext() {
    getContext().makeContext(nullptr);
}

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
    if (glfwWindowShouldClose(mWindow))
        return false;
    return true;
}

void GLWindow::resize(const uvec2 size) {
    glfwSetWindowSize(mWindow, size.x,size.y);
}

uvec2 GLWindow::size() const {
    int w, h;
    glfwGetFramebufferSize(mWindow, &w, &h);
    return { w,h };
}

GLFWwindow* GLWindow::get() const {
    return mWindow;
}

GLWindow::~GLWindow() {
    glDeleteFramebuffers(1, &mFBO);
    glfwDestroyWindow(mWindow);
}

GLIMGUIWindow::GLIMGUIWindow() {
    if (!ImGui_ImplGlfwGL3_Init(mWindow,true))
        throw std::runtime_error("Failed to setup ImGui binding.");
}

void GLIMGUIWindow::newFrame() {
    makeContext();
    ImGui_ImplGlfwGL3_NewFrame();
}

void GLIMGUIWindow::renderGUI() {
    makeContext();
    ImGui::Render();
}

GLIMGUIWindow::~GLIMGUIWindow() {
    makeContext();
    ImGui_ImplGlfwGL3_Shutdown();
}

GLImage::GLImage():mRes(nullptr) {
    glGenTextures(1, &mTexture);
}

GLImage::~GLImage() {
    if(mRes)checkError(cudaGraphicsUnregisterResource(mRes));
    glDeleteTextures(1, &mTexture);
}

uvec2 GLImage::size() const {
    return mSize;
}

void GLImage::resize(const uvec2 size) {
    if (mSize != size) {
        if (mRes) {
            checkError(cudaGraphicsUnregisterResource(mRes));
            mRes = nullptr;
        }
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, size.x, size.y
            , 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        checkError(cudaGraphicsGLRegisterImage(&mRes, mTexture, GL_TEXTURE_2D
            , cudaGraphicsRegisterFlagsSurfaceLoadStore));
        mSize = size;
    }
}

cudaArray_t GLImage::bind(const cudaStream_t stream) {
    checkError(cudaGraphicsMapResources(1, &mRes, stream));
    cudaArray_t data;
    checkError(cudaGraphicsSubResourceGetMappedArray(&data, mRes, 0, 0));
    return data;
}

void GLImage::unbind(const cudaStream_t stream) {
    checkError(cudaGraphicsUnmapResources(1, &mRes, stream));
}

GLuint GLImage::get() const {
    return mTexture;
}

