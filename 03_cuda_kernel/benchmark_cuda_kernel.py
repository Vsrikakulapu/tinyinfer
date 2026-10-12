"""
TinyInfer step 3: benchmark a custom fused bias-add + ReLU CUDA kernel
against PyTorch's native (unfused) equivalent, across several tensor shapes.
Verifies numerical correctness (max abs error) alongside the speedup.
"""
import torch
import torch.nn.functional as F
import time
from torch.utils.cpp_extension import load

print("Compiling fused_bias_relu CUDA extension (first run only)...", flush=True)
fused_bias_relu_ext = load(
    name="fused_bias_relu_ext",
    sources=["fused_bias_relu.cu"],
    verbose=False,
)


def native_bias_relu(x, bias):
    return F.relu(x + bias.view(1, -1, 1, 1))


def benchmark(fn, *args, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iters * 1e6  # microseconds


shapes = [
    (1, 32, 32, 32),
    (1, 64, 32, 32),
    (1, 128, 16, 16),
    (1, 256, 8, 8),
    (32, 128, 16, 16),
]

header = "{:<20}{:<14}{:<14}{:<10}{:<12}".format("shape", "native (us)", "fused (us)", "speedup", "max_abs_err")
print(header)
for shape in shapes:
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, device="cuda")
    bias = torch.randn(C, device="cuda")

    out_native = native_bias_relu(x, bias)
    out_fused = fused_bias_relu_ext.fused_bias_relu(x, bias)
    max_err = (out_native - out_fused).abs().max().item()

    t_native = benchmark(native_bias_relu, x, bias)
    t_fused = benchmark(fused_bias_relu_ext.fused_bias_relu, x, bias)
    speedup = t_native / t_fused

    print(f"{str(shape):<20}{t_native:<14.2f}{t_fused:<14.2f}{speedup:<10.2f}{max_err:<12.2e}")
