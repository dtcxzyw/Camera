#include <Base/Pipeline.hpp>

Pipeline::Pipeline() {
    checkError(cudaStreamCreate(&mStream));
}

Pipeline::~Pipeline() {
    checkError(cudaStreamDestroy(mStream));
}

void Pipeline::sync() {
    checkError(cudaStreamSynchronize(mStream));
}

cudaStream_t Pipeline::getId() const {
    return mStream;
}
