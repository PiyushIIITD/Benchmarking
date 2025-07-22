#include "common.h" 
#define CUTENSOR_ALIGNED_UNIT_BYTE 1

extern cutensorHandle_t g_cutensorHandle;
void createCutensorTensorDescriptor(cutensorTensorDescriptor_t& desc, size_t N, cutensorDataType_t dtype) {
    int64_t extent = static_cast<int64_t>(N);
    int64_t stride = 1;
    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(g_cutensorHandle, &desc, 1, &extent, &stride, dtype, CUTENSOR_ALIGNED_UNIT_BYTE));
}
void relu_cutensor(float* d_in, float* d_out, size_t N) {
    cutensorTensorDescriptor_t in_desc, out_desc;
    createCutensorTensorDescriptor(in_desc, N, CUTENSOR_R_32F);
    createCutensorTensorDescriptor(out_desc, N, CUTENSOR_R_32F);

    CUTENSOR_CHECK(cutensorElementwiseUnary(
        g_cutensorHandle,
        CUTENSOR_OP_RELU,
        in_desc, d_in,
        out_desc, d_out,
        nullptr
    ));

    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(in_desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(out_desc));
}
void linear_cutensor(float* d_in, float* d_out, size_t N) {
    CUDA_CHECK(cudaMemcpy(d_out, d_in, N * sizeof(float), cudaMemcpyDeviceToDevice));
}
void sigmoid_cutensor(float* d_in, float* d_out, size_t N) {
    cutensorTensorDescriptor_t in_desc, out_desc;
    createCutensorTensorDescriptor(in_desc, N, CUTENSOR_R_32F);
    createCutensorTensorDescriptor(out_desc, N, CUTENSOR_R_32F);

    CUTENSOR_CHECK(cutensorElementwiseUnary(
        g_cutensorHandle,
        CUTENSOR_OP_SIGMOID,
        in_desc, d_in,
        out_desc, d_out,
        nullptr
    ));

    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(in_desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(out_desc));
}
void tanh_cutensor(float* d_in, float* d_out, size_t N) {
    cutensorTensorDescriptor_t in_desc, out_desc;
    createCutensorTensorDescriptor(in_desc, N, CUTENSOR_R_32F);
    createCutensorTensorDescriptor(out_desc, N, CUTENSOR_R_32F);

    CUTENSOR_CHECK(cutensorElementwiseUnary(
        g_cutensorHandle,
        CUTENSOR_OP_TANH,
        in_desc, d_in,
        out_desc, d_out,
        nullptr
    ));

    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(in_desc));
    CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(out_desc));
}

// ---------------- Softmax ----------------
// NOTE: cuTENSOR doesn’t have direct softmax. You would implement it via exp → sum → divide.
// For now, fallback to CUDA softmax implementation.
void softmax_cutensor(float* d_in, float* d_out, size_t N) {
    std::cout << "cuTENSOR doesn't support softmax directly. Falling back to custom CUDA kernel.\n";
    softmax_cuda(d_in, d_out, N);  // Use your own CUDA version here
}
