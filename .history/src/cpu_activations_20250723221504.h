#pragma once
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <algorithm>

void relu_cpu(const float* in, float* out, size_t N);
void linear_cpu(const float* in, float* out, size_t N);
void sigmoid_cpu(const float* in, float* out, size_t N);
void tanh_cpu(const float* in, float* out, size_t N);
void softmax_cpu(const float* in, float* out, size_t N);