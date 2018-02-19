#include <Base/Pipeline.hpp>
#include <stdexcept>

Stream::Stream() {
    checkError(cudaStreamCreateWithFlags(&mStream,cudaStreamNonBlocking));
    mMaxThread = getEnvironment().getProp().maxThreadsPerBlock;
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

void Environment::init() {
    int cnt;
    checkError(cudaGetDeviceCount(&cnt));
    cudaDeviceProp prop{};
    auto id = -1, maxwell = 0;
    size_t size = 0;
    for (auto i = 0; i < cnt; ++i) {
        checkError(cudaGetDeviceProperties(&prop, i));
        const auto ver = prop.major * 10000 + prop.minor;
        if (maxwell < ver || (maxwell==ver && prop.totalGlobalMem>size))
            id = i, maxwell = ver,size=prop.totalGlobalMem;
    }
    if (id == -1)
        throw std::runtime_error("Failed to initialize the CUDA environment.");
    checkError(cudaSetDevice(id));
    checkError(cudaGetDeviceProperties(&mProp, id));
}

const cudaDeviceProp& Environment::getProp() const {
    return mProp;
}

Environment::~Environment() {
    checkError(cudaDeviceSynchronize());
    checkError(cudaDeviceReset());
}

Environment& getEnvironment() {
    static Environment env;
    return env;
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
