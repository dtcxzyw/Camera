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
    SwapChain(size_t size) {
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
