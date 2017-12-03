#include <GL/glew.h>
#include <Interaction/OpenGL.hpp>
#include <IMGUI/imgui_impl_glfw_gl3.h>
#include <exception>

class GLContext final:Singletion {
private:
    bool mFlag;
    GLContext():mFlag(false) {
        if (!glfwInit())
            throw std::exception("Failed to initialize glfw.");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }
    friend GLContext& getContext();
public:
    void makeContext(GLFWwindow* window) {
        GLFWwindow* current = nullptr;
        if (current != window) {
            glfwMakeContextCurrent(window);
            glDisable(GL_FRAMEBUFFER_SRGB);
            current = window;
            if (!mFlag) {
                glewExperimental = true;
                if (glewInit() != GLEW_NO_ERROR)
                    throw std::exception("Failed to initialize glew.");
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
        throw std::exception("Failed to create a window.");
    context.makeContext(mWindow);
    glfwSwapInterval(0);
    glGenFramebuffers(1, &mFBO);
}

void GLWindow::present(SharedImage image) {
    getContext().makeContext(mWindow);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFBO);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D
        , image->get(), 0);
    auto isiz = image->size();
    auto frame = size();
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, frame.x, frame.y, 0, 0, isiz.x, isiz.y
        , GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void GLWindow::swapBuffers() {
    glfwSwapBuffers(mWindow);
}

bool GLWindow::update() {
    glfwPollEvents();
    if (glfwWindowShouldClose(mWindow))
        return false;
    return true;
}

void GLWindow::resize(size_t width, size_t height) {
    glfwSetWindowSize(mWindow, width, height);
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
        throw std::exception("Failed to setup ImGui binding.");
}

void IMGUIWindow::newFrame() {
    getContext().makeContext(mWindow);
    ImGui_ImplGlfwGL3_NewFrame();
}

IMGUIWindow::~IMGUIWindow() {
    getContext().makeContext(mWindow);
    ImGui_ImplGlfwGL3_Shutdown();
}

Image::Image():mRes(0) {
    glGenTextures(1, &mTexture);
}

Image::~Image() {
    if(mRes)checkError(cudaGraphicsUnregisterResource(mRes));
    glDeleteTextures(1, &mTexture);
}

uvec2 Image::size() const {
    return mSize;
}

void Image::resize(uvec2 size) {
    if (mSize != size) {
        if (mRes) {
            checkError(cudaGraphicsUnregisterResource(mRes));
            mRes = 0;
        }
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y
            , 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        checkError(cudaGraphicsGLRegisterImage(&mRes, mTexture, GL_TEXTURE_2D
            , cudaGraphicsRegisterFlagsSurfaceLoadStore));
        mSize = size;
    }
}

cudaArray_t Image::bind(Stream & stream) {
    checkError(cudaGraphicsMapResources(1, &mRes, stream.getID()));
    cudaArray_t data;
    checkError(cudaGraphicsSubResourceGetMappedArray(&data, mRes, 0, 0));
    return data;
}

void Image::unbind(Stream & stream) {
    checkError(cudaGraphicsUnmapResources(1, &mRes, stream.getID()));
}

GLuint Image::get() const {
    return mTexture;
}

SwapChain::SwapChain(size_t size) {
    for (size_t i = 0; i < size; ++i)
        mImages.emplace_back(std::make_shared<Image>());
}

SharedImage SwapChain::pop() {
    if (mImages.empty())return nullptr;
    auto res = mImages.back();
    mImages.pop_back();
    return res;
}

void SwapChain::push(SharedImage image) {
    mImages.emplace_back(image);
}
