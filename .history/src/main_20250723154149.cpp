#include "common.h" 
#include <iostream>
#include <vector>
#include <iomanip> 
#include "cuda_activations.cuh"
#include "cpu_activations.cpp"
#include "/home/piyush/Benchmarking/src/cuda_activations.cu"
#include "cudnn_activations.cpp"
// #include "cutensor_activations.cu"

cudnnHandle_t g_cudnnHandle;
cutensorHandle_t g_cutensorHandle;

/*************  ✨ Windsurf Command ⭐  *************/
/**
 * @brief Run a benchmark for a given activation function.
 *
 * @param name the name of the activation function
 * @param num_elements the number of elements in the input and output arrays

/*******  1cb25087-2eb3-479b-9104-3eddb2d2062e  *******/
void run_benchmark(const std::string& name, size_t num_elements, float alpha_val = 0.01f) {
    std::cout << "\n Benchmarking " << name << " (N=" << num_elements << ")" << std::endl;
    std::vector<float> h_in(num_elements);
    std::vector<float> h_out_cpu(num_elements);
    std::vector<float> h_out_gpu(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        h_in[i] = (float)rand() / RAND_MAX * 100.0f - 50.0f; 
    }
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
    const int WARMUP_RUNS = 10;
    const int BENCHMARK_RUNS = 100;

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (name == "ReLU") relu_cpu(h_in.data(), h_out_cpu.data(), num_elements);
        else if (name == "Linear") linear_cpu(h_in.data(), h_out_cpu.data(), num_elements);
        else if (name == "Sigmoid") sigmoid_cpu(h_in.data(), h_out_cpu.data(), num_elements);
        else if (name == "Tanh") tanh_cpu(h_in.data(), h_out_cpu.data(), num_elements);
        else if (name == "Softmax") softmax_cpu(h_in.data(), h_out_cpu.data(), num_elements);
    }
    double cpu_time_ms = 0.0;
    cpu_time_ms = measureCPUTime([&]() {
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            if (name == "ReLU") relu_cpu(h_in.data(), h_out_cpu.data(), num_elements);
            else if (name == "Linear") linear_cpu(h_in.data(), h_out_cpu.data(), num_elements);
            else if (name == "Sigmoid") sigmoid_cpu(h_in.data(), h_out_cpu.data(), num_elements);
            else if (name == "Tanh") tanh_cpu(h_in.data(), h_out_cpu.data(), num_elements);
            else if (name == "Softmax") softmax_cpu(h_in.data(), h_out_cpu.data(), num_elements);
        }
    });
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU Time: " << cpu_time_ms / BENCHMARK_RUNS << " ms/op" << std::endl;

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (name == "ReLU") relu_cuda(d_in, d_out, num_elements);
        else if (name == "Linear") linear_cuda(d_in, d_out, num_elements);
        else if (name == "Sigmoid") sigmoid_cuda(d_in, d_out, num_elements);
        else if (name == "Tanh") tanh_cuda(d_in, d_out, num_elements);
        else if (name == "Softplus") softmax_cuda(d_in, d_out, num_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float custom_cuda_time_ms = 0.0f;
    for (int i = 0; i < BENCHMARK_RUNS; ++i) {
        custom_cuda_time_ms += measureGPUTime(
            [&](float* d_i, float* d_o, size_t n) {
                if (name == "ReLU") relu_cuda(d_i, d_o, n);
                else if (name == "Linear") linear_cuda(d_i, d_o, n);
                else if (name == "Sigmoid") sigmoid_cuda(d_i, d_o, n);
                else if (name == "Tanh") tanh_cuda(d_i, d_o, n);
                else if (name == "Softmax") softmax_cuda(d_i, d_o, n);
            }, d_in, d_out, num_elements
        );
    }
    std::cout << "Custom CUDA Time: " << custom_cuda_time_ms / BENCHMARK_RUNS << " ms/op" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResults(h_out_cpu.data(), h_out_gpu.data(), num_elements);

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (name == "ReLU") relu_cudnn(d_in, d_out, num_elements);
        else if (name == "Leaky ReLU") linear_cudnn(d_in, d_out, num_elements);
        else if (name == "Sigmoid") sigmoid_cudnn(d_in, d_out, num_elements);
        else if (name == "Tanh") tanh_cudnn(d_in, d_out, num_elements);
        else if (name == "Softplus") softmax_cudnn(d_in, d_out, num_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float cudnn_time_ms = 0.0f;
    for (int i = 0; i < BENCHMARK_RUNS; ++i) {
        cudnn_time_ms += measureGPUTime(
            [&](float* d_i, float* d_o, size_t n) {
                if (name == "ReLU") relu_cudnn(d_i, d_o, n);
                else if (name == "Linear") linear_cudnn(d_i, d_o, n);
                else if (name == "Sigmoid") sigmoid_cudnn(d_i, d_o, n);
                else if (name == "Tanh") tanh_cudnn(d_i, d_o, n);
                else if (name == "Softmax") softmax_cudnn(d_i, d_o, n); 
            }, d_in, d_out, num_elements
        );
    }
    std::cout << "cuDNN Time: " << cudnn_time_ms / BENCHMARK_RUNS << " ms/op" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResults(h_out_cpu.data(), h_out_gpu.data(), num_elements);

    // for (int i = 0; i < WARMUP_RUNS; ++i) {
    //     if (name == "ReLU") relu_cutensor(d_in, d_out, num_elements);
    //     else if (name == "Linear") linear_cutensor(d_in, d_out, num_elements);
    //     else if (name == "Sigmoid") sigmoid_cutensor(d_in, d_out, num_elements);
    //     else if (name == "Tanh") tanh_cutensor(d_in, d_out, num_elements);
    //     else if (name == "Softmax") softmax_cutensor(d_in, d_out, num_elements); 
    //     CUDA_CHECK(cudaDeviceSynchronize());
    // }
    // float cutensor_time_ms = 0.0f;
    // for (int i = 0; i < BENCHMARK_RUNS; ++i) {
    //     cutensor_time_ms += measureGPUTime(
    //         [&](float* d_i, float* d_o, size_t n) {
    //             if (name == "ReLU") relu_cutensor(d_i, d_o, n);
    //             else if (name == "Leaky ReLU") linear_cutensor(d_i, d_o, n);
    //             else if (name == "Sigmoid") sigmoid_cutensor(d_i, d_o, n);
    //             else if (name == "Tanh") tanh_cutensor(d_i, d_o, n);
    //             else if (name == "Softplus") softmax_cutensor(d_i, d_o, n);
    //         }, d_in, d_out, num_elements
    //     );
    // }
    // std::cout << "cuTENSOR Time: " << cutensor_time_ms / BENCHMARK_RUNS << " ms/op" << std::endl;
    // CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    // verifyResults(h_out_cpu.data(), h_out_gpu.data(), num_elements);
    // CUDA_CHECK(cudaFree(d_in));
    // CUDA_CHECK(cudaFree(d_out));
}

int main() {
    CUDNN_CHECK(cudnnCreate(&g_cudnnHandle));
    CUTENSOR_CHECK(cutensorCreate(&g_cutensorHandle));
    srand(static_cast<unsigned int>(time(NULL)));
    std::vector<size_t> sizes = {1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024}; 
    for (size_t N : sizes) {
        run_benchmark("ReLU", N);
        run_benchmark("Linear", N, 0.1f);
        run_benchmark("Sigmoid", N);
        run_benchmark("Tanh", N);
        run_benchmark("Softmax", N);
    }
    CUDNN_CHECK(cudnnDestroy(g_cudnnHandle));
    CUTENSOR_CHECK(cutensorDestroy(g_cutensorHandle));

    return 0;
}