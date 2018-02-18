#include <Base/CompileBegin.hpp>
#include <GL/glew.h>
#include <Interaction/OpenGL.hpp>
#include <IMGUI/imgui_impl_glfw_gl3.h>
#include <Base/CompileEnd.hpp>

class GLContext final:Singletion {
private:
    bool mFlag;
    GLContext():mFlag(false) {
        if (!glfwInit())
            throw std::runtime_error("Failed to initialize glfw.");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }
    friend GLContext& getContext();
public:
    void makeContext(GLFWwindow* window) {
        static GLFWwindow* current = nullptr;
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

GLContext& getContext() {
    static GLContext context;
    return context;
}

GLWindow::GLWindow() {
    auto& context=getContext();
    mWindow = glfwCreateWindow(800, 600, "OpenGL Viewer", nullptr, nullptr);
    if (!mWindow)
        throw std::runtime_error("Failed to create a window.");
    context.makeContext(mWindow);
    glfwSwapInterval(0);
    glGenFramebuffers(1, &mFBO);
}

void GLWindow::present(Image& image) {
    getContext().makeContext(mWindow);
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
    getContext().makeContext(mWindow);
    glfwSwapInterval(enable);
}

void GLWindow::swapBuffers() {
    glfwSwapBuffers(mWindow);
}

bool GLWindow::update() {
    getContext().makeContext(mWindow);
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

GLWindow::~GLWindow() {
    glDeleteFramebuffers(1, &mFBO);
    glfwDestroyWindow(mWindow);
}

IMGUIWindow::IMGUIWindow() {
    if (!ImGui_ImplGlfwGL3_Init(mWindow,true))
        throw std::runtime_error("Failed to setup ImGui binding.");
}

void IMGUIWindow::newFrame() {
    getContext().makeContext(mWindow);
    ImGui_ImplGlfwGL3_NewFrame();
}

void IMGUIWindow::renderGUI() {
    getContext().makeContext(mWindow);
    const auto wsiz = size();
    glViewport(0, 0, wsiz.x, wsiz.y);
    ImGui::Render();
}

IMGUIWindow::~IMGUIWindow() {
    getContext().makeContext(mWindow);
    ImGui_ImplGlfwGL3_Shutdown();
}

Image::Image():mRes(nullptr) {
    glGenTextures(1, &mTexture);
}

Image::~Image() {
    if(mRes)checkError(cudaGraphicsUnregisterResource(mRes));
    glDeleteTextures(1, &mTexture);
}

uvec2 Image::size() const {
    return mSize;
}

void Image::resize(const uvec2 size) {
    if (mSize != size) {
        if (mRes) {
            checkError(cudaGraphicsUnregisterResource(mRes));
            mRes = nullptr;
        }
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y
            , 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        checkError(cudaGraphicsGLRegisterImage(&mRes, mTexture, GL_TEXTURE_2D
            , cudaGraphicsRegisterFlagsSurfaceLoadStore));
        mSize = size;
    }
}

cudaArray_t Image::bind(const cudaStream_t stream) {
    checkError(cudaGraphicsMapResources(1, &mRes, stream));
    cudaArray_t data;
    checkError(cudaGraphicsSubResourceGetMappedArray(&data, mRes, 0, 0));
    return data;
}

void Image::unbind(const cudaStream_t stream) {
    checkError(cudaGraphicsUnmapResources(1, &mRes, stream));
}

GLuint Image::get() const {
    return mTexture;
}

