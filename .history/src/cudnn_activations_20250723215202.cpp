#include "commoner.h"
#include "common.h"
extern cudnnHandle_t g_cudnnHandle;
void createTensorDescriptor(cudnnTensorDescriptor_t& descriptor, size_t N) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        static_cast<int>(N), 1, 1, 1 
    ));
}
void relu_cudnn(float* d_in, float* d_out, size_t N) {
    cudnnTensorDescriptor_t x_desc, y_desc;
    createTensorDescriptor(x_desc, N);
    createTensorDescriptor(y_desc, N);

    cudnnActivationDescriptor_t activation_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(g_cudnnHandle, activation_desc, &alpha, x_desc, d_in, &beta, y_desc, d_out));

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}
void linear_cudnn(float* d_in, float* d_out, size_t N) {
    CUDA_CHECK(cudaMemcpy(d_out, d_in, N * sizeof(float), cudaMemcpyDeviceToDevice));
}
void sigmoid_cudnn(float* d_in, float* d_out, size_t N) {
    cudnnTensorDescriptor_t x_desc, y_desc;
    createTensorDescriptor(x_desc, N);
    createTensorDescriptor(y_desc, N);

    cudnnActivationDescriptor_t activation_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(g_cudnnHandle, activation_desc, &alpha, x_desc, d_in, &beta, y_desc, d_out));

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}
void tanh_cudnn(float* d_in, float* d_out, size_t N) {
    cudnnTensorDescriptor_t x_desc, y_desc;
    createTensorDescriptor(x_desc, N);
    createTensorDescriptor(y_desc, N);

    cudnnActivationDescriptor_t activation_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(g_cudnnHandle, activation_desc, &alpha, x_desc, d_in, &beta, y_desc, d_out));

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}

void softmax_cudnn(float* d_in, float* d_out, size_t N) {
    cudnnTensorDescriptor_t input_desc, output_desc;
    createTensorDescriptor(input_desc, N);
    createTensorDescriptor(output_desc, N);

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CHECK(cudnnSoftmaxForward(
        g_cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        input_desc,
        d_in,
        &beta,
        output_desc,
        d_out
    ));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
}