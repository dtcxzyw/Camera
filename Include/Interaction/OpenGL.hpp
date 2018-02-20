#pragma once
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <GLFW/glfw3.h>
#include <Base/CompileEnd.hpp>
#include <Base/Builtin.hpp>
#include <Base/Environment.hpp>

class GLImage final :Uncopyable {
private:
    GLuint mTexture;
    cudaGraphicsResource_t mRes;
    uvec2 mSize;
public:
    GLImage();
    ~GLImage();
    uvec2 size() const;
    void resize(uvec2 size);
    cudaArray_t bind(cudaStream_t stream);
    void unbind(cudaStream_t stream);
    GLuint get() const;
};

class GLWindow:Uncopyable {
protected:
    GLFWwindow* mWindow;
    GLuint mFBO;
public:
    explicit GLWindow(GLWindow* share = nullptr);
    void makeContext();
    void unmakeContext();
    void present(GLImage& image);
    void setVSync(bool enable);
    void swapBuffers();
    bool update();
    void resize(uvec2 size);
    uvec2 size() const;
    GLFWwindow* get() const;
    ~GLWindow();
};

class GLIMGUIWindow final :public GLWindow {
public:
    GLIMGUIWindow();
    void newFrame();
    void renderGUI();
    ~GLIMGUIWindow();
};
