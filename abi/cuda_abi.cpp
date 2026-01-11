#include "cuda_abi.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" void launch_add_kernel(float* a, float* b, float* c, int n);

// Internal Error Handling

static cudaError_t last_error = cudaSuccess;

static void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        last_error = err;
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Memory Management

extern "C" void* cuda_malloc(size_t bytes) {
    void* ptr = nullptr; 
    cudaError_t err = cudaMalloc(&ptr, bytes); 
    check_cuda_error(err); 

    if (err != cudaSuccess) {
        return nullptr;
    }
    
    return ptr; 
}

extern "C" void cuda_free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    cudaError_t err = cudaFree(ptr); 
    check_cuda_error(err); 
}

extern "C" void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    check_cuda_error(err);
}

extern "C" void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
}

// Error Checking

extern "C" const char* cuda_get_last_error() {
    if (last_error == cudaSuccess) {
        return "Success";
    }
    return cudaGetErrorString(last_error); 
}

extern "C" int cuda_synchronize() {
    cudaError_t err = cudaDeviceSynchronize(); 
    check_cuda_error(err); 

    return (err == cudaSuccess) ? 0 : -1; 
}

// Operations

extern "C" void cuda_add(float* a, float* b, float* c, int n) {
    launch_add_kernel(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    check_cuda_error(err);
}