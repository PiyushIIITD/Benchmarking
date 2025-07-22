#include "common.h"
#include <cuda_runtime.h>
#include <cutensor.h>
#include <vector>
#include <math.h>
#include <assert.h>

#define CHECK_CUTENSOR(err) do { \
    auto status = (err); \
    if (status != CUTENSOR_STATUS_SUCCESS) { \
        printf("cuTENSOR error %d: %s\n", status, cutensorGetErrorString(status)); \
        exit(1); \
    } \
} while (0)

extern cutensorHandle_t g_cutensorHandle;

// CUDA kernels for pointwise ops
__global__ void relu_kernel(float* in, float* out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = fmaxf(in[i], 0.0f);
}

__global__ void sigmoid_kernel(float* in, float* out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

__global__ void tanh_kernel(float* in, float* out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = tanhf(in[i]);
}

// Softmax (on host)
void softmax_cutensor(float* d_in, float* d_out, size_t N) {
    float* h = new float[N];
    CUDA_CHECK(cudaMemcpy(h, d_in, N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_val = h[0];
    for (size_t i = 1; i < N; ++i)
        if (h[i] > max_val) max_val = h[i];
    float sum = 0;
    for (size_t i = 0; i < N; ++i) {
        h[i] = expf(h[i] - max_val);
        sum += h[i];
    }
    for (size_t i = 0; i < N; ++i) h[i] /= sum;
    CUDA_CHECK(cudaMemcpy(d_out, h, N * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h;
}

void relu_cutensor(float* d_in, float* d_out, size_t N) {
    relu_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
}

void sigmoid_cutensor(float* d_in, float* d_out, size_t N) {
    sigmoid_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
}

void tanh_cutensor(float* d_in, float* d_out, size_t N) {
    tanh_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
}

// Linear: simulate y = Wx with cuTENSOR contraction
void linear_cutensor(float* d_in, float* d_out, size_t N) {
    // We'll simulate y = Wx where W is identity
    // i.e., W: [N, N], x: [N], y: [N]

    float* d_W;
    CUDA_CHECK(cudaMalloc(&d_W, N * N * sizeof(float)));

    // Fill W with identity
    std::vector<float> h_W(N * N, 0);
    for (size_t i = 0; i < N; ++i)
        h_W[i * N + i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    cutensorTensorDescriptor_t descA, descB, descC;
    int64_t modeA[2] = {'i', 'k'}; // W
    int64_t modeB[1] = {'k'};      // x
    int64_t modeC[1] = {'i'};      // y

    int64_t extentA[2] = {static_cast<int64_t>(N), static_cast<int64_t>(N)};
    int64_t extentB[1] = {static_cast<int64_t>(N)};
    int64_t extentC[1] = {static_cast<int64_t>(N)};

    CHECK_CUTENSOR(cutensorInitTensorDescriptor(&g_cutensorHandle, &descA, 2, extentA, NULL, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY));
    CHECK_CUTENSOR(cutensorInitTensorDescriptor(&g_cutensorHandle, &descB, 1, extentB, NULL, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY));
    CHECK_CUTENSOR(cutensorInitTensorDescriptor(&g_cutensorHandle, &descC, 1, extentC, NULL, CUTENSOR_R_32F, CUTENSOR_OP_IDENTITY));

    cutensorContractionDescriptor_t desc;
    CHECK_CUTENSOR(cutensorInitContractionDescriptor(
        &g_cutensorHandle, &desc, 
        &descA, modeA,  // A = W
        &descB, modeB,  // B = x
        &descC, modeC,  // C = y
        &descC, modeC,  // D = y
        CUTENSOR_R_32F
    ));

    cutensorContractionFind_t find;
    CHECK_CUTENSOR(cutensorInitContractionFind(&g_cutensorHandle, &find, CUTENSOR_ALGO_DEFAULT));

    size_t workspace_size;
    CHECK_CUTENSOR(cutensorContractionGetWorkspace(&g_cutensorHandle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0)
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    cutensorContractionPlan_t plan;
    CHECK_CUTENSOR(cutensorInitContractionPlan(&g_cutensorHandle, &plan, &desc, &find, workspace_size));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUTENSOR(cutensorContraction(&g_cutensorHandle, &plan, &alpha, d_W, d_in, &beta, d_out, d_out, workspace, workspace_size));

    CUDA_CHECK(cudaFree(d_W));
    if (workspace) CUDA_CHECK(cudaFree(workspace));
}
