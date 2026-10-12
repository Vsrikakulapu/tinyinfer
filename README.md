# TinyInfer

I built TinyInfer to learn what actually makes a model faster at inference time. I trained a
small CNN, then compared INT8 quantization, ONNX Runtime and a custom CUDA kernel against plain
PyTorch.

The model is a MobileNet-style network with about 137K parameters, trained on CIFAR-10. I kept
it small so everything runs in a few minutes on a free Colab T4.

PyTorch, CUDA C++, ONNX Runtime.

## Results

From one run on a Colab T4, latency per image at batch size 1:

| Setup                       | Accuracy | Size (MB) | Latency (ms) |
|-----------------------------|----------|-----------|--------------|
| FP32 PyTorch on GPU         | 71.67%   | 0.58      | 1.420        |
| INT8 PyTorch on CPU         | 73.30%*  | 0.57      | 2.575        |
| FP32 ONNX Runtime on CPU    | -        | -         | 0.351        |

\* INT8 accuracy is on the first 1,000 test images because the quantized model is slow to
evaluate on CPU. The FP32 number uses all 10,000, so they aren't directly comparable.

ONNX Runtime on CPU ended up about 4x faster than PyTorch on the GPU and 7x faster than the INT8
model. For a model this small at batch size 1, most of the time goes to per-operation overhead
(Python dispatch, kernel launches), not actual math, and ONNX Runtime's optimized graph cuts most
of that out.

Quantization barely changed anything, and it took me a bit to see why: PyTorch's dynamic
quantization only converts Linear layers, so only the final classifier was quantized and all the
convolutions stayed FP32. Getting real gains would need static quantization with calibration.

## Custom CUDA kernel

Every conv layer in the model is followed by a bias add and a ReLU, which PyTorch runs as two
separate kernels. I wrote a CUDA kernel that does both in one pass and loaded it into PyTorch as
a C++/CUDA extension.

| Shape (N, C, H, W) | PyTorch (us) | Fused (us) | Speedup |
|--------------------|--------------|------------|---------|
| (1, 32, 32, 32)    | 32.44        | 17.84      | 1.82x   |
| (1, 64, 32, 32)    | 31.57        | 17.73      | 1.78x   |
| (1, 128, 16, 16)   | 31.24        | 18.95      | 1.65x   |
| (1, 256, 8, 8)     | 29.00        | 14.57      | 1.99x   |
| (32, 128, 16, 16)  | 80.42        | 51.73      | 1.55x   |

The output matches PyTorch exactly on every shape. Times are averaged over 100 runs after 10
warm-up runs. The speedup mostly comes from launching one kernel instead of two and not writing
an intermediate tensor to memory and reading it back. At batch size 1 the launch overhead is a
big share of the time, which is why the gain is larger there than at batch 32.

## Running it

You need a CUDA GPU (Colab works). Run the scripts in order:

```bash
python 01_train_baseline.py
python 02_quantize_export.py
cd 03_cuda_kernel && python benchmark_cuda_kernel.py
```

## What I'd try next

Static quantization of the conv layers, running ONNX Runtime and TensorRT on the GPU, testing
larger batch sizes, and fusing the bias and ReLU directly into the convolution.
