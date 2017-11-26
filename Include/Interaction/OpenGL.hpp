#pragma once
#include <IMGUI/imgui.h>
#include <GLFW/glfw3.h>
#include <Base/Builtin.hpp>
#include <cuda_gl_interop.h>

class GLWindow:Uncopyable {
protected:
    GLFWwindow* mWindow;
    GLuint mFBO, mTexture;
    cudaGraphicsResource_t mRes;
    uvec2 mSize;
    void resize(uvec2 size);
public:
    GLWindow();
    void present(Stream& stream,const BuiltinRenderTarget<RGBA>& colorBuffer);
    cudaArray_t map(Stream& stream, uvec2 size);
    void unmapAndPresent(Stream& stream);
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
