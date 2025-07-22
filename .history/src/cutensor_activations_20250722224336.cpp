#include "common.h" 
#include <cutensor/types.h>
#include <cutensor/reduction.h>
#include <cutensor/operations.h>
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

void softmax_cutensor(float* d_in, float* d_out, size_t N) {
    float* d_exp;
    CUDA_CHECK(cudaMalloc(&d_exp, N * sizeof(float)));
    {
        cutensorTensorDescriptor_t in_desc, exp_desc;
        createCutensorTensorDescriptor(in_desc, N, CUTENSOR_R_32F);
        createCutensorTensorDescriptor(exp_desc, N, CUTENSOR_R_32F);

        CUTENSOR_CHECK(cutensorElementwiseUnary(
            g_cutensorHandle,
            CUTENSOR_OP_EXP,
            in_desc, d_in,
            exp_desc, d_exp,
            nullptr
        ));

        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(in_desc));
        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(exp_desc));
    }
    float sum_exp = 0.0f;
    {
        cutensorTensorDescriptor_t exp_desc;
        createCutensorTensorDescriptor(exp_desc, N, CUTENSOR_R_32F);

        CUTENSOR_CHECK(cutensorReduction(
            g_cutensorHandle,
            CUTENSOR_OP_ADD,
            exp_desc, d_exp,    
            1.0f,             // alpha
            1.0f,             // beta
            nullptr,             
            0,                   
            CUTENSOR_R_32F,
            &sum_exp,            
            nullptr              
        ));

        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(exp_desc));
    }

    {
        cutensorTensorDescriptor_t exp_desc, out_desc;
        createCutensorTensorDescriptor(exp_desc, N, CUTENSOR_R_32F);
        createCutensorTensorDescriptor(out_desc, N, CUTENSOR_R_32F);

        float inv_sum = 1.0f / sum_exp;

        CUTENSOR_CHECK(cutensorElementwiseBinary(
            g_cutensorHandle,
            CUTENSOR_OP_MUL,
            exp_desc, d_exp,
            nullptr, nullptr,  
            out_desc, d_out,
            &inv_sum  
        ));

        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(exp_desc));
        CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(out_desc));
    }

    CUDA_CHECK(cudaFree(d_exp));
}