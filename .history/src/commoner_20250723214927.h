#pragma once

#include <cuda_runtime.h>
#include <cstddef>

void initializeDeviceMemory(float* d_ptr, size_t N);
void verifyResults(const float* A, const float* B, size_t N, float tol = 1e-5f);

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    }
