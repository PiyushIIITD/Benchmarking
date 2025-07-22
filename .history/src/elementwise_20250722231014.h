#ifndef CUTENSOR_ELEMENTWISE_H
#define CUTENSOR_ELEMENTWISE_H

#include <cutensor.h>

void relu_cutensor(float* d_input, float* d_output, size_t num_elements);
void linear_cutensor(float* d_input, float* d_output, size_t num_elements);
void sigmoid_cutensor(float* d_input, float* d_output, size_t num_elements);
void tanh_cutensor(float* d_input, float* d_output, size_t num_elements);
void softmax_cutensor(float* d_input, float* d_output, size_t num_elements);

#endif
