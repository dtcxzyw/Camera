#include <Interaction/OpenGL.hpp>
#include <exception>

class GLContext final:Singletion {
private:
    GLContext() {
        if (!glfwInit())
            throw std::exception("Failed to initialize glfw.");
    }
    friend GLContext& getContext();
public:
    void makeContext(GLFWwindow* window) {
        glfwMakeContextCurrent(window);
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
    context.makeContext(mWindow);
}

void GLWindow::present(Pipeline& pipeline,const BuiltinRenderTarget<RGBA>& colorbuffer) {
    getContext().makeContext(mWindow);
    auto data=colorbuffer.download(pipeline);
    pipeline.sync();
    auto size = colorbuffer.size();
    glDrawPixels(size.x,size.y,GL_RGBA,GL_FLOAT,data.begin());
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
    glfwDestroyWindow(mWindow);
}
