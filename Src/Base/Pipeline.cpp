#include <Base/Pipeline.hpp>

Stream::Stream() {
    checkError(cudaStreamCreateWithFlags(&mStream,cudaStreamNonBlocking));
    int cur;
    checkError(cudaGetDevice(&cur));
    cudaDeviceProp prop{};
    checkError(cudaGetDeviceProperties(&prop,cur));
    mMaxThread = prop.maxThreadsPerBlock;
}

Stream::~Stream() {
    checkError(cudaStreamDestroy(mStream));
}

void Stream::sync() {
    checkError(cudaStreamSynchronize(mStream));
}

cudaStream_t Stream::get() const {
    return mStream;
}

cudaError_t Stream::query() const {
    return cudaStreamQuery(mStream);
}

void Stream::wait(Event & event) {
    checkError(cudaStreamWaitEvent(mStream,event.get(),0));
}

Event::Event(Stream& stream) {
    checkError(cudaEventCreateWithFlags(&mEvent, cudaEventDisableTiming));
    checkError(cudaEventRecord(mEvent, stream.get()));
}

void Event::wait() {
    checkError(cudaEventSynchronize(mEvent));
}

cudaEvent_t Event::get() const {
    return mEvent;
}

Event::~Event() {
    checkError(cudaEventDestroy(mEvent));
}
