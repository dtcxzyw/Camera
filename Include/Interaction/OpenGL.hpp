#pragma once
#include <Core/Config.hpp>
#ifdef CAMERA_OPENGL_SUPPORT
#include <Core/Common.hpp>
#include <Math/Math.hpp>
#include <Interaction/Counter.hpp>

class GLFWwindow;

class GLWindow final : public Singletion<GLWindow> {
private:
    GLFWwindow* mWindow;
    uint32_t mFBO;
    float mWheel;
    bool mPressed[3]{};
    Counter mCounter;

    friend void mouseButtonCallback(GLFWwindow*, int button, int action, int);
    friend void scrollCallback(GLFWwindow*, double, double y);
    void makeContext() const;
public:
    void enumDevices(int* buf, uint32_t* count) const;

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
#endif
