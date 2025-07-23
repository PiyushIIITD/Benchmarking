#ifndef CUDA_ACTIVATIONS_CUH
#define CUDA_ACTIVATIONS_CUH
#include <cstddef>
void relu_cuda(float* d_in, float* d_out, size_t N);
void linear_cuda(float* d_in, float* d_out, size_t N);
void sigmoid_cuda(float* d_in, float* d_out, size_t N);
void tanh_cuda(float* d_in, float* d_out, size_t N);
void softmax_cuda(float* d_in, float* d_out, size_t N);

#endif
