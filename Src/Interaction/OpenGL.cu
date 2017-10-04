#include <GL/glew.h>
#include <Interaction/OpenGL.hpp>
#include <exception>

class GLContext final:Singletion {
private:
    bool mFlag;
    GLContext():mFlag(false) {
        if (!glfwInit())
            throw std::exception("Failed to initialize glfw.");
    }
    friend GLContext& getContext();
public:
    void makeContext(GLFWwindow* window) {
        GLFWwindow* current = nullptr;
        if (current != window) {
            glfwMakeContextCurrent(window);
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

void GLWindow::resize(uvec2 size) {
    if (size != mSize) {
        if (mRes) { 
            checkError(cudaGraphicsUnregisterResource(mRes));
            mRes = 0;
        }
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,size.x,size.y
            ,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
        checkError(cudaGraphicsGLRegisterImage(&mRes, mTexture, GL_TEXTURE_2D
            , cudaGraphicsRegisterFlagsSurfaceLoadStore));
        mSize = size;
    }
}

GLWindow::GLWindow():mRes(0) {
    auto& context=getContext();
    mWindow = glfwCreateWindow(800, 600, "OpenGL Viewer", nullptr, nullptr);
    if (!mWindow)
        throw std::exception("Failed to create a window.");
    context.makeContext(mWindow);
    glfwSwapInterval(0);
    glGenTextures(1, &mTexture);
    resize(size());
    glGenFramebuffers(1, &mFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D
        , mTexture, 0);
    auto res = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    if(res!=GL_FRAMEBUFFER_COMPLETE)
        throw std::exception("Failed to create a FBO.");
}

void GLWindow::present(Pipeline& pipeline,const BuiltinRenderTarget<RGBA>& colorbuffer) {
    getContext().makeContext(mWindow);
    resize(colorbuffer.size());
    checkError(cudaGraphicsMapResources(1, &mRes, pipeline.getId()));
    cudaArray_t data;
    checkError(cudaGraphicsSubResourceGetMappedArray(&data, mRes, 0, 0));
    checkError(cudaMemcpyArrayToArray(data, 0, 0, colorbuffer.get()
        , 0, 0, mSize.x*mSize.y * sizeof(RGBA)));
    checkError(cudaGraphicsUnmapResources(1, &mRes, pipeline.getId()));
    auto frame = size();
    glBindFramebuffer(GL_READ_FRAMEBUFFER,mFBO);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0,0,frame.x,frame.y,0,0,mSize.x,mSize.y
        ,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
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
    if(mRes)checkError(cudaGraphicsUnregisterResource(mRes));
    glDeleteTextures(1, &mTexture);
    glfwDestroyWindow(mWindow);
}
