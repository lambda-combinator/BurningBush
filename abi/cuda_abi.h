#ifndef CUDA_ABI_H
#define CUDA_ABI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory Management

void* cuda_malloc(size_t size); 

void cuda_free(void* ptr); 

void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes); 

void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);

// Error Checking

const char* cuda_get_last_error();

int cuda_synchronize(); 

// Operations

void cuda_add(float* a, float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif