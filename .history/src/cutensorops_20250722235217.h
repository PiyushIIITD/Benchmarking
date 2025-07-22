#pragma once
// #include "common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

#ifndef CUTENSOR_CHECK
#define CUTENSOR_CHECK(status) \
    if ((status) != 0) throw std::runtime_error("cuTENSOR operation failed")
#endif

#define CUTENSOR_OP_RELU     1
#define CUTENSOR_OP_TANH     2
#define CUTENSOR_OP_SIGMOID  3
#define CUTENSOR_OP_EXP      4
#define CUTENSOR_OP_ADD      5
#define CUTENSOR_OP_MUL      6

#define CUTENSOR_R_32F       0
#define CUTENSOR_ALIGNED_UNIT_BYTE 1

typedef int cutensorHandle_t;
typedef int cutensorTensorDescriptor_t;
typedef int cutensorDataType_t;

inline int cutensorCreate(cutensorHandle_t* handle) {
    *handle = 1;
    return 0;
}

inline int cutensorDestroy(cutensorHandle_t) {
    return 0;
}

inline int cutensorCreateTensorDescriptor(
    cutensorHandle_t, cutensorTensorDescriptor_t* desc, int, const int64_t*, const int64_t*,
    cutensorDataType_t, uint32_t) {
    *desc = 1;
    return 0;
}

inline int cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t) {
    return 0;
}

inline int cutensorElementwiseUnary(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_in,
    cutensorTensorDescriptor_t, float* d_out, void*) {
    
    float h_in, h_out;
    cudaMemcpy(&h_in, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    switch (op) {
        case CUTENSOR_OP_RELU: h_out = fmaxf(0, h_in); break;
        case CUTENSOR_OP_TANH: h_out = tanhf(h_in); break;
        case CUTENSOR_OP_EXP:  h_out = expf(h_in); break;
        default: h_out = h_in; break;
    }

    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
    return 0;
}

inline int cutensorElementwiseBinary(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_A,
    void*, void*, cutensorTensorDescriptor_t, float* d_out, const float* scalar) {

    float h_A, h_out;
    float h_scalar = scalar ? scalar[0] : 1.0f;

    cudaMemcpy(&h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);

    switch (op) {
        case CUTENSOR_OP_MUL: h_out = h_A * h_scalar; break;
        default: h_out = h_A; break;
    }

    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
    return 0;
}

inline int cutensorElementwiseTrinary(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_in,
    cutensorTensorDescriptor_t, float* d_out, void*) {

    float h_in, h_out;
    cudaMemcpy(&h_in, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    if (op == CUTENSOR_OP_SIGMOID)
        h_out = 1.0f / (1.0f + expf(-h_in));
    else
        h_out = h_in;

    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
    return 0;
}

inline int cutensorReduction(
    cutensorHandle_t, int op, cutensorTensorDescriptor_t, const float* d_in,
    float alpha, float beta, void*, int, cutensorDataType_t, float* h_out, void*) {

    float h_val;
    cudaMemcpy(&h_val, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    switch (op) {
        case CUTENSOR_OP_ADD: *h_out = h_val * alpha + beta; break;
        default: *h_out = h_val; break;
    }

    return 0;
}
