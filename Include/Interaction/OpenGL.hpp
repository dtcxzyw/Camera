#pragma once
#include <Base/CompileBegin.hpp>
#include <GLFW/glfw3.h>
#include <Base/CompileEnd.hpp>
#include <Base/Common.hpp>
#include <Base/Math.hpp>
#include <Interaction/Counter.hpp>

class GLWindow final:public Singletion<GLWindow> {
private:
    GLFWwindow* mWindow;
    GLuint mFBO;
    float mWheel;
    bool mPressed[3]{};
    Counter mCounter;

    friend void mouseButtonCallback(GLFWwindow*, int button, int action, int);
    friend void scrollCallback(GLFWwindow*, double, double y);
    void makeContext();
public:
    GLWindow();
    //void present(GLImage& image);
    void newFrame();
    void setVSync(bool enable);
    void swapBuffers();
    bool update();
    void resize(uvec2 size);
    uvec2 size() const;
    ~GLWindow();
};
