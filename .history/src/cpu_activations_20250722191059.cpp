// activations_cpu.cpp

#include "common.h"  // Use relative path in your project structure
#include <cmath>
#include <algorithm>
#include <numeric>   // for std::accumulate

// ReLU activation: max(0, x)
void relu_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

// Linear activation: identity
void linear_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = in[i];
    }
}

// Sigmoid activation: 1 / (1 + exp(-x))
void sigmoid_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

// Tanh activation: tanh(x)
void tanh_cpu(const float* in, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

// Softmax activation: exp(x_i) / sum(exp(x_j))
void softmax_cpu(const float* in, float* out, size_t N) {
    float max_val = *std::max_element(in, in + N); // for numerical stability
    float sum = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }

    for (size_t i = 0; i < N; ++i) {
        out[i] /= sum;
    }
}
