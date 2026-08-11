#pragma once
#include <cstddef>
namespace nvinfer1 {
    enum class Severity { kINTERNAL_ERROR = 0, kERROR = 1, kWARNING = 2, kINFO = 3, kVERBOSE = 4 };
    class ILogger { public: virtual void log(Severity severity, const char* msg) noexcept = 0; };
    class IExecutionContext {};
    class ICudaEngine { public: virtual IExecutionContext* createExecutionContext() noexcept = 0; };
    class IRuntime { public: virtual ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size) noexcept = 0; };
    IRuntime* createInferRuntime(ILogger& logger) noexcept;
}
