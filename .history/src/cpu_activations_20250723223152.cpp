// #include "commoner.h"  
#include <bits/stdc++.h>
#include <cmath>
#include <algorithm>
#include <numeric> 

void relu_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

void linear_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = in[i];
    }
}

void sigmoid_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void tanh_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

void softmax_cpu(const float* in, float* out, size_t N) {
    float max_val = *std::max_element(in, in + N);
    float sum = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }

    for (size_t i = 0; i < N; ++i) {
        out[i] /= sum;
    }
}
