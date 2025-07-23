#pragma once
#include <bits/stdc++.h>
void relu_cudnn(float* d_in, float* d_out, size_t N);
void linear_cudnn(float* d_in, float* d_out, size_t N);
void sigmoid_cudnn(float* d_in, float* d_out, size_t N);
void tanh_cudnn(float* d_in, float* d_out, size_t N);
void softmax_cudnn(float* d_in, float* d_out, size_t N);