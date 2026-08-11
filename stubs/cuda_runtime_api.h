#pragma once
#include <cstddef>
typedef int cudaError_t;
cudaError_t cudaMalloc(void** devPtr, std::size_t size);
cudaError_t cudaFree(void* devPtr);
