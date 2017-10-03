#pragma once
#include <GLFW/glfw3.h>
#include <Base/Builtin.hpp>

class GLWindow {
protected:
    GLFWwindow* mWindow;
public:
    GLWindow();
    GLWindow(const GLWindow&) = delete;
    GLWindow& operator=(const GLWindow&) = delete;
    void present(Pipeline& pipeline,const BuiltinRenderTarget<RGBA>& colorbuffer);
    bool update();
    void resize(size_t width, size_t height);
    uvec2 size() const;
    ~GLWindow();
};
