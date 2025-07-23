#include "common.h"
#include <algorithm>
#include <iostream>

void initializeDeviceMemory(float* d_ptr, size_t N) {
    std::vector<float> h(N);
    for(size_t i = 0; i < N; ++i)
        h[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f - 50.0f;
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
}

void verifyResults(const float* A, const float* B, size_t N, float tol) {
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(A[i] - B[i]) > tol) {
            std::cerr << "Mismatch at " << i << ": " << A[i] << " vs " << B[i] << "\n";
            exit(1);
        }
    }
    std::cout << "Results match ✔\n";
}
