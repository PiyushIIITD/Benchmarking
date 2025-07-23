#include <bits/stdc++.h>
#include "common.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>
#include <functional>
#include <cassert>


#define THREADS_PER_BLOCK 256
__global__ void relu_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void sigmoid_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

__global__ void tanh_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = tanhf(input[idx]);
}

__global__ void linear_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx];
}

__global__ void softmax_kernel(float* input, float* output, size_t size) {
    float max_val = -INFINITY;
    for (int i = 0; i < size; ++i) {
        max_val = fmaxf(max_val, input[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += expf(input[i] - max_val);
    }
    for (int i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

extern "C" void relu_cuda(float* input, float* output, size_t size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

extern "C" void sigmoid_cuda(float* input, float* output, size_t size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sigmoid_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

extern "C" void tanh_cuda(float* input, float* output, size_t size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    tanh_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

extern "C" void linear_cuda(float* input, float* output, size_t size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    linear_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

extern "C" void softmax_cuda(float* input, float* output, size_t size) {
    softmax_kernel<<<1, 1>>>(input, output, size);
    cudaDeviceSynchronize();
}
