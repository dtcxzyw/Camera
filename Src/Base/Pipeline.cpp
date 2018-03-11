#include <Base/Pipeline.hpp>
#include <Base/Environment.hpp>

Stream::Stream() {
    checkError(cudaStreamCreateWithFlags(&mStream,cudaStreamNonBlocking));
    mMaxThread = getDeviceMonitor().getProp().maxThreadsPerBlock;
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

Event::Event() {
    checkError(cudaEventCreateWithFlags(&mEvent, cudaEventDisableTiming));
}

void Event::bind(Stream& stream) {
    checkError(cudaEventRecord(mEvent, stream.get()));
}

void Event::wait() {
    checkError(cudaEventSynchronize(mEvent));
}

cudaError_t Event::query() const {
    return cudaEventQuery(mEvent);
}

cudaEvent_t Event::get() const {
    return mEvent;
}

Event::~Event() {
    checkError(cudaEventDestroy(mEvent));
}
