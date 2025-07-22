#pragma once
#include "common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#define CUTENSOR_CHECK(status) \
    if ((status) != 0) throw std::runtime_error("cuTENSOR operation failed")
#define CUTENSOR_OP_RELU     1
#define CUTENSOR_OP_TANH     2
#define CUTENSOR_OP_SIGMOID  3
#define CUTENSOR_OP_EXP      4
#define CUTENSOR_OP_ADD      5
#define CUTENSOR_OP_MUL      6

using cutensorHandle_t = int;
using cutensorTensorDescriptor_t = int;
using cutensorDataType_t = int;
#define CUTENSOR_R_32F 0
#define CUTENSOR_ALIGNED_UNIT_BYTE 1

inline int cutensorCreate(cutensorHandle_t* handle) {
    *handle = 1; return 0;
}

inline int cutensorDestroy(cutensorHandle_t handle) {
    return 0;
}

inline int cutensorCreateTensorDescriptor(
    cutensorHandle_t, cutensorTensorDescriptor_t* desc, int, const int64_t*, const int64_t*,
    cutensorDataType_t, uint32_t) {
   *desc = 1; return 0;
}

inline int cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t) { return 0; }

inline int cutensorElementwiseUnary(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_in,
    cutensorTensorDescriptor_t, float* d_out, void*) {

    float* h_in = new float[1];
    cudaMemcpy(h_in, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    float h_out = 0;
    if (op == CUTENSOR_OP_RELU) h_out = fmaxf(0, h_in[0]);
    else if (op == CUTENSOR_OP_TANH) h_out = tanhf(h_in[0]);
    else if (op == CUTENSOR_OP_EXP) h_out = expf(h_in[0]);
    else h_out = h_in[0];

    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_in;
    return 0;
}

inline int cutensorElementwiseBinary(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_A,
    void*, void*, cutensorTensorDescriptor_t, float* d_out, const float* scalar) {

    float* h_A = new float[1];
    float h_scalar = scalar ? scalar[0] : 1.0f;

    cudaMemcpy(h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    float result = (op == CUTENSOR_OP_MUL) ? h_A[0] * h_scalar : h_A[0];
    cudaMemcpy(d_out, &result, sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_A;
    return 0;
}

inline int cutensorElementwiseTrinary(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_in,
    cutensorTensorDescriptor_t, float* d_out, void*) {
    float* h_in = new float[1];
    cudaMemcpy(h_in, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    float h_out = 0;
    if (op == CUTENSOR_OP_SIGMOID)
        h_out = 1.0f / (1.0f + expf(-h_in[0]));
    else
        h_out = h_in[0];
    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_in;
    return 0;
}

inline int cutensorReduction(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_in,
    float alpha, float beta, void*, int, cutensorDataType_t, float* h_out, void*) {
    float h_val;
    cudaMemcpy(&h_val, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    *h_out = h_val * alpha + beta;
    return 0;
}
