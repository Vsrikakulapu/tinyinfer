"""
TinyInfer step 4: collect results from steps 2 and 3 into one summary.
Run after 02_quantize_export.py has produced quantize_export_results.json.
(Copy the printed table from benchmark_cuda_kernel.py's output manually,
or extend it to also dump JSON if you want this fully automated.)
"""
import json

with open("quantize_export_results.json") as f:
    results = json.load(f)

fp32 = results["fp32"]
int8 = results["int8_dynamic"]
onnx = results["onnx_runtime_cpu"]

print("=" * 60)
print("TinyInfer — Inference Optimization Summary")
print("=" * 60)
print(f"{'Config':<20}{'Accuracy':<12}{'Size (MB)':<12}{'Latency (ms)'}")
print(f"{'FP32 (GPU)':<20}{fp32['acc']:<12.4f}{fp32['size_mb']:<12.2f}{fp32['latency_ms']:.3f}")
print(f"{'INT8 dynamic (CPU)':<20}{int8['acc']:<12.4f}{int8['size_mb']:<12.2f}{int8['latency_ms']:.3f}")
if onnx["latency_ms"]:
    print(f"{'ONNX Runtime (CPU)':<20}{'--':<12}{'--':<12}{onnx['latency_ms']:.3f}")

size_reduction = (1 - int8["size_mb"] / fp32["size_mb"]) * 100
acc_delta = (int8["acc"] - fp32["acc"]) * 100
print()
print(f"INT8 quantization: {size_reduction:.1f}% smaller, {acc_delta:+.2f}pp accuracy change")
print()
print("(Append the custom CUDA kernel speedup numbers from")
print(" 03_cuda_kernel/benchmark_cuda_kernel.py's printed table here.)")
