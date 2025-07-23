#pragma once
#include <bits/stdc++.h>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cutensor.h>
#include <functional>
#include <cmath>
#include <cstdlib>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudnnGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUTENSOR_CHECK(call) \
    do { \
        cutensorStatus_t status = call; \
        if (status != CUTENSOR_STATUS_SUCCESS) { \
            std::cerr << "cuTENSOR Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cutensorGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

inline float measureGPUTime(const std::function<void(float*, float*, size_t)>&func, float* d_in, float* d_out, size_t num_elements) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    func(d_in, d_out, num_elements);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds;
}

inline double measureCPUTime(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void initializeDeviceMemory(float* d_ptr, size_t num_elements) {
    std::vector<float> h_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
}

bool verifyResults(const float* h_out_cpu, const float* h_out_gpu, size_t num_elements, float epsilon = 1e-4) {
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(h_out_cpu[i] - h_out_gpu[i]) > epsilon) {
            std::cerr << "Verification failed at index " << i << ": CPU=" << h_out_cpu[i]
                      << ", GPU=" << h_out_gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

struct CuDNNHandle {
    cudnnHandle_t handle;
    CuDNNHandle()  { CUDNN_CHECK(cudnnCreate(&handle)); }
    ~CuDNNHandle() { cudnnDestroy(handle); }
    cudnnHandle_t get() const { return handle; }
};

struct CuTensorHandle {
    cutensorHandle_t handle;
    CuTensorHandle()  {
        CUTENSOR_CHECK(cutensorCreate(&handle)); 
    }
    ~CuTensorHandle() {  }
    cutensorHandle_t get() const { return handle; }
};

inline float* allocateDeviceMemory(size_t num_elements) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, num_elements * sizeof(float)));
    return d_ptr;
}

inline float* allocateHostMemory(size_t num_elements) {
    return new float[num_elements];
}

inline void freeMemory(float* h_ptr, float* d_ptr) {
    delete[] h_ptr;
    CUDA_CHECK(cudaFree(d_ptr));
}

// void relu_cuda(float* d_in, float* d_out, size_t N);
// void linear_cuda(float* d_in, float* d_out, size_t N);
// void sigmoid_cuda(float* d_in, float* d_out, size_t N);
// void tanh_cuda(float* d_in, float* d_out, size_t N);
// void softmax_cuda(float* d_in, float* d_out, size_t N);

void relu_cudnn(float* d_in, float* d_out, size_t N);
void linear_cudnn(float* d_in, float* d_out, size_t N);
void sigmoid_cudnn(float* d_in, float* d_out, size_t N);
void tanh_cudnn(float* d_in, float* d_out, size_t N);
void softmax_cudnn(float* d_in, float* d_out, size_t N);

// void relu_cutensor(float* d_in, float* d_out, size_t N);
// void linear_cutensor(float* d_in, float* d_out, size_t N);
// void sigmoid_cutensor(float* d_in, float* d_out, size_t N);
// void tanh_cutensor(float* d_in, float* d_out, size_t N);
// void softmax_cutensor(float* d_in, float* d_out, size_t N);
