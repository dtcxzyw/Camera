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

void Stream::wait(Event& event) {
    checkError(cudaStreamWaitEvent(mStream, event.get(), 0));
}

Event::Event(const bool recordTime) {
    auto flag = cudaEventDefault;
    if (!recordTime) {
        flag |= cudaEventDisableTiming;
        if (Environment::get().getAppType() == AppType::Offline)
            flag |= cudaEventBlockingSync;
    }
    checkError(cudaEventCreateWithFlags(&mEvent, flag));
}

void Event::bind(Stream& stream) {
    checkError(cudaEventRecord(mEvent, stream.get()));
}

void Event::sync() {
    checkError(cudaEventSynchronize(mEvent));
}

cudaError_t Event::query() const {
    return cudaEventQuery(mEvent);
}

cudaEvent_t Event::get() const {
    return mEvent;
}

float Event::operator-(Event& rhs) const {
    float res;
    checkError(cudaEventElapsedTime(&res, rhs.get(), get()));
    return res;
}

Event::~Event() {
    checkError(cudaEventDestroy(mEvent));
}
