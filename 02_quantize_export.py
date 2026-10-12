"""
TinyInfer step 2: post-training INT8 quantization + ONNX export, with a
latency/throughput/size benchmark across FP32 PyTorch, INT8 PyTorch, and
ONNX Runtime.

Requires baseline_fp32.pt from 01_train_baseline.py.
"""
import os
import time
import json
import torch
import torch.nn as nn

import importlib.util
spec = importlib.util.spec_from_file_location("baseline_mod", "01_train_baseline.py")
baseline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_mod)
TinyMobileNet = baseline_mod.TinyMobileNet
get_loaders = baseline_mod.get_loaders
evaluate = baseline_mod.evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_size_mb(model):
    torch.save(model.state_dict(), "_tmp_size.pt")
    size = os.path.getsize("_tmp_size.pt") / 1e6
    os.remove("_tmp_size.pt")
    return size


def benchmark_latency(fn, input_tensor, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        fn(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        fn(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - t0
    return (total / n_iters) * 1000


def main():
    print("Loading data...", flush=True)
    _, test_loader = get_loaders()

    print("Evaluating FP32 baseline...", flush=True)
    fp32_model = TinyMobileNet().to(device)
    fp32_model.load_state_dict(torch.load("baseline_fp32.pt", map_location=device))
    fp32_model.eval()

    fp32_acc = evaluate(fp32_model, test_loader)
    fp32_size = model_size_mb(fp32_model)

    dummy = torch.randn(1, 3, 32, 32, device=device)
    fp32_latency = benchmark_latency(lambda x: fp32_model(x), dummy)

    print(f"[FP32]  acc={fp32_acc:.4f}  size={fp32_size:.2f}MB  latency={fp32_latency:.3f}ms", flush=True)

    print("Quantizing to INT8 (CPU)...", flush=True)
    cpu_model = TinyMobileNet()
    cpu_model.load_state_dict(torch.load("baseline_fp32.pt", map_location="cpu"))
    cpu_model.eval()

    int8_model = torch.quantization.quantize_dynamic(
        cpu_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    int8_size = model_size_mb(int8_model)
    print(f"Quantized. Size on disk: {int8_size:.2f}MB. Evaluating on a 1000-image subset...", flush=True)

    subset = torch.utils.data.Subset(test_loader.dataset, range(1000))
    cpu_test_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

    def evaluate_cpu(model, loader):
        # Local CPU-only eval — the imported evaluate() from step 1 always
        # moves tensors to the module-level `device` (cuda here), which
        # breaks a CPU-only quantized model. Keep everything on CPU explicitly.
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                print(f"  eval batch {i+1}/{len(loader)}...", flush=True)
        return correct / total

    int8_acc = evaluate_cpu(int8_model, cpu_test_loader)
    dummy_cpu = torch.randn(1, 3, 32, 32)
    int8_latency = benchmark_latency(lambda x: int8_model(x), dummy_cpu, n_warmup=3, n_iters=20)

    print(f"[INT8]  acc={int8_acc:.4f} (on 1000-image subset)  size={int8_size:.2f}MB  latency={int8_latency:.3f}ms (CPU)", flush=True)

    fp32_model.eval()
    torch.onnx.export(
        fp32_model.to("cpu"), dummy_cpu, "model_fp32.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
    )
    print("Exported model_fp32.onnx")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("model_fp32.onnx", providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        def onnx_infer(x):
            return sess.run(None, {input_name: x.numpy()})

        onnx_latency = benchmark_latency(onnx_infer, dummy_cpu)
        print(f"[ONNX Runtime CPU]  latency={onnx_latency:.3f}ms")
    except ImportError:
        onnx_latency = None
        print("onnxruntime not installed")

    results = {
        "fp32": {"acc": fp32_acc, "size_mb": fp32_size, "latency_ms": fp32_latency, "device": str(device)},
        "int8_dynamic": {"acc": int8_acc, "size_mb": int8_size, "latency_ms": int8_latency, "device": "cpu"},
        "onnx_runtime_cpu": {"latency_ms": onnx_latency},
    }
    with open("quantize_export_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved quantize_export_results.json")


if __name__ == "__main__":
    main()
