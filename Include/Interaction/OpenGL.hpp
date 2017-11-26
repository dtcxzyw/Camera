#pragma once
#include <IMGUI/imgui.h>
#include <GLFW/glfw3.h>
#include <Base/Builtin.hpp>
#include <cuda_gl_interop.h>

class Image final :Uncopyable {
private:
    GLuint mTexture;
    cudaGraphicsResource_t mRes;
    uvec2 mSize;
public:
    Image();
    ~Image();
    uvec2 size() const;
    void resize(uvec2 size);
    cudaArray_t bind(Stream& stream);
    void unbind(Stream& stream);
    GLuint get() const;
};

using SharedImage = std::shared_ptr<Image>;

class SwapChain final:Uncopyable {
private:
    std::vector<SharedImage> mImages;
public:
    SwapChain(size_t size);
    SharedImage pop();
    void push(SharedImage image);
};

class GLWindow:Uncopyable {
protected:
    GLFWwindow* mWindow;
    GLuint mFBO;
public:
    GLWindow();
    void present(SharedImage image);
    void swapBuffers();
    bool update();
    void resize(size_t width, size_t height);
    uvec2 size() const;
    ~GLWindow();
};

class IMGUIWindow final :public GLWindow {
public:
    IMGUIWindow();
    void newFrame();
    void renderGUI() {
        auto wsiz = size();
        glViewport(0, 0, wsiz.x, wsiz.y);
        ImGui::Render();
    }
    ~IMGUIWindow();
};
