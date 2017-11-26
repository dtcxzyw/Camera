#include <Base/Pipeline.hpp>

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

cudaStream_t Stream::getId() const {
    return mStream;
}

cudaError_t Stream::query() const {
    return cudaStreamQuery(mStream);
}

void Stream::wait(Event & event) {
    checkError(cudaStreamWaitEvent(mStream,event.get(),0));
}

Environment::Environment(){}

void Environment::init() {
    int cnt;
    checkError(cudaGetDeviceCount(&cnt));
    cudaDeviceProp prop;
    int id = -1, maxwell = 0;
    size_t size = 0;
    for (int i = 0; i < cnt; ++i) {
        checkError(cudaGetDeviceProperties(&prop, i));
        int ver = prop.major * 10000 + prop.minor;
        if (ver < 50002)continue;
        if (!prop.unifiedAddressing)continue;
        if (!prop.managedMemory)continue;
        if (maxwell < ver || (maxwell==ver && prop.totalGlobalMem>size))
            id = i, maxwell = ver,size=prop.totalGlobalMem;
    }
    if (id == -1)
        throw std::exception("Failed to initialize the CUDA environment.");
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
    checkError(cudaEventRecord(mEvent, stream.getId()));
}

void Event::wait() {
    checkError(cudaEventSynchronize(mEvent));
}

cudaEvent_t Event::get() {
    return mEvent;
}

Event::~Event() {
    checkError(cudaEventDestroy(mEvent));
}
