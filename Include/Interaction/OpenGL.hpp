#pragma once
#include <Base/CompileBegin.hpp>
#include <IMGUI/imgui.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <Base/CompileEnd.hpp>
#include <Base/Builtin.hpp>

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
    cudaArray_t bind(cudaStream_t stream);
    void unbind(cudaStream_t stream);
    GLuint get() const;
};

template<typename Frame>
class SwapChain final:Uncopyable {
private:
    std::vector<std::shared_ptr<Frame>> mImages;
public:
    using SharedFrame = std::shared_ptr<Frame>;
    SwapChain(const size_t size) {
        for (size_t i = 0; i < size; ++i)
            mImages.emplace_back(std::make_shared<Frame>());
    }
    SharedFrame pop() {
        if (mImages.empty())return nullptr;
        auto ptr = std::move(mImages.back());
        mImages.pop_back();
        return ptr;
    }
    void push(SharedFrame&& image) {
        mImages.emplace_back(std::move(image));
    }
    bool empty() const {
        return mImages.empty();
    }
};

class GLWindow:Uncopyable {
protected:
    GLFWwindow* mWindow;
    GLuint mFBO;
public:
    GLWindow();
    void present(Image& image);
    void setVSync(bool enable);
    void swapBuffers();
    bool update();
    void resize(uvec2 size);
    uvec2 size() const;
    ~GLWindow();
};

class IMGUIWindow final :public GLWindow {
public:
    IMGUIWindow();
    void newFrame();
    void renderGUI();
    ~IMGUIWindow();
};
