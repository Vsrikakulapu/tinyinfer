#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(const float* x, const float* bias, float* out, int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx < total) {
        int c = (idx / HW) % C;
        float val = x[idx] + bias[c];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor fused_bias_relu(torch::Tensor x, torch::Tensor bias) {
    auto out = torch::empty_like(x);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int HW = H * W;
    int total = N * C * HW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_bias_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), N, C, HW
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_relu", &fused_bias_relu, "Fused bias-add + ReLU (CUDA)");
}
