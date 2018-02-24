#pragma once
#include <Base/CompileBegin.hpp>
#include <GLFW/glfw3.h>
#include <Base/CompileEnd.hpp>
#include <Interaction/BoundImage.hpp>

class GLImage final :public BoundImage {
private:
    GLuint mTexture;
    void reset() override;
public:
    GLImage();
    ~GLImage();
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
