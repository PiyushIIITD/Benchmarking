#include "common.h"
#include <cfloat>
#define THREADS_PER_BLOCK 256

__global__ void relu_kernel(const float* in, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = max(0.0f, in[idx]);
    }
}

void relu_cuda(float* d_in, float* d_out, size_t N) {
    dim3 dimGrid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 dimBlock(THREADS_PER_BLOCK);
    relu_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void linear_kernel(const float* in, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];
    }
}
void linear_cuda(float* d_in, float* d_out, size_t N) {
    dim3 dimGrid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 dimBlock(THREADS_PER_BLOCK);
    linear_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void sigmoid_kernel(const float* in, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

void sigmoid_cuda(float* d_in, float* d_out, size_t N) {
    dim3 dimGrid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 dimBlock(THREADS_PER_BLOCK);
    sigmoid_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void tanh_kernel(const float* in, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = tanhf(in[idx]);
    }
}

void tanh_cuda(float* d_in, float* d_out, size_t N) {
    dim3 dimGrid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 dimBlock(THREADS_PER_BLOCK);
    tanh_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void softmax_kernel(const float* in, float* out, size_t N) {
    __shared__ float max_val;
    __shared__ float sum;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, in[i]);
    }
    if (threadIdx.x == 0) max_val = -FLT_MAX;
    __syncthreads();
    atomicMax((int*)&max_val, __float_as_int(local_max));
    __syncthreads();
    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float val = expf(in[i] - max_val);
        local_sum += val;
        out[i] = val;
    }
    if (threadIdx.x == 0) sum = 0.0f;
    __syncthreads();
    atomicAdd(&sum, local_sum);
    __syncthreads();
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        out[i] /= sum;
    }
}

void softmax_cuda(float* d_in, float* d_out, size_t N) {
    dim3 dimGrid(1); 
    dim3 dimBlock(THREADS_PER_BLOCK);
    softmax_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
