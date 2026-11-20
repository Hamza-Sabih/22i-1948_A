"""Deliverable 11 - automated benchmarking of every trained model.

For each model (teacher, 3 baseline students, 3 distilled students, NAS-best):
  parameters, checkpoint size, ONNX size, FLOPs, test accuracy (PyTorch),
  ONNX Runtime CPU accuracy, PyTorch CPU latency (b=1), ONNX Runtime CPU latency (b=1),
  ONNX Runtime throughput (b=64), peak inference memory (CPU, and GPU when available).
Writes results/benchmark_results.csv, results/kd_comparison.csv and all plots.

Usage:  python -m src.deploy.benchmark --threads 1
"""
import argparse

import numpy as np
import pandas as pd

from ..common.data import load_cifar
from ..common.profiling import (count_flops, gpu_peak_memory_mb, ort_cpu_latency, ort_session,
                                peak_inference_memory_mb, torch_cpu_latency)
from ..common.utils import RESULTS_DIR, count_params, load_json, setup_logging, write_csv
from .onnx_inference import OnnxClassifier, evaluate_onnx
from .registry import available_keys, ckpt_path, describe, load_model, onnx_path


def main():
    p = argparse.ArgumentParser(description="Benchmark all trained models")
    p.add_argument("--threads", type=int, default=1, help="CPU threads (1 = edge-like single core)")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--n-test", type=int, default=10000, help="test images for ONNX CPU accuracy")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    setup_logging("benchmark")
    data_cache, rows = {}, []
    for key in available_keys(include_teacher=True):
        model, meta = load_model(key)
        res = meta["resolution"]
        shape = (1, 3, res, res)
        info = describe(key)
        row = {"key": key, **info, "params_M": count_params(model) / 1e6,
               "ckpt_size_mb": ckpt_path(key).stat().st_size / 2 ** 20,
               "mflops": (count_flops(model, shape) or 0) / 1e6, "input_res": res,
               "torch_top1": meta.get("test_top1"), "torch_top5": meta.get("test_top5")}
        lat = torch_cpu_latency(model, shape, warmup=10, runs=max(20, args.runs // 2 if key == "resnet50_teacher" else args.runs),
                                threads=args.threads)
        row["torch_cpu_latency_ms"] = lat["median_ms"]
        mem = peak_inference_memory_mb(model, shape)
        row["peak_cpu_mem_mb"] = mem["peak_total_mb"] if mem else None
        row["peak_gpu_mem_mb"] = gpu_peak_memory_mb(model, shape)
        op = onnx_path(key)
        if op.exists():
            ds = info["dataset"]
            if ds not in data_cache:
                data_cache[ds] = load_cifar(ds, smoke=args.smoke)
            d = data_cache[ds]
            n = min(args.n_test, len(d["y_test"]))
            clf = OnnxClassifier(key, threads=args.threads)
            ev = evaluate_onnx(clf, d["x_test"][:n], d["y_test"][:n])
            sess = ort_session(op, args.threads)
            l1 = ort_cpu_latency(sess, shape, runs=args.runs)
            l64 = ort_cpu_latency(sess, (64, 3, res, res), warmup=3, runs=10)
            row.update({"onnx_size_mb": op.stat().st_size / 2 ** 20, "onnx_cpu_top1": ev["top1"],
                        "onnx_cpu_top5": ev["top5"], "onnx_eval_images": n,
                        "ort_cpu_latency_ms": l1["median_ms"], "ort_cpu_latency_p90_ms": l1["p90_ms"],
                        "ort_throughput_b64_img_s": 64 / (l64["median_ms"] / 1e3),
                        "ort_speedup_vs_torch": lat["median_ms"] / l1["median_ms"]})
        row["cpu_threads"] = args.threads
        rows.append(row)
        print(f"[bench] {key:28s} params={row['params_M']:.3f}M size={row['ckpt_size_mb']:.2f}MB "
              f"MFLOPs={row['mflops']:.1f} top1={row['torch_top1']} "
              f"torch-cpu={row['torch_cpu_latency_ms']:.2f}ms ort-cpu={row.get('ort_cpu_latency_ms', float('nan')):.2f}ms")
    write_csv(rows, RESULTS_DIR / "benchmark_results.csv")

    # KD vs baseline comparison table
    df = pd.DataFrame(rows)
    comp = []
    for m in df[df.variant == "baseline"].model:
        b = df[(df.model == m) & (df.variant == "baseline")].iloc[0]
        k = df[(df.model == m) & (df.variant == "kd")]
        if len(k):
            k = k.iloc[0]
            comp.append({"student": m, "params_M": b.params_M, "baseline_top1": b.torch_top1, "kd_top1": k.torch_top1,
                         "kd_gain_top1": k.torch_top1 - b.torch_top1, "baseline_top5": b.torch_top5,
                         "kd_top5": k.torch_top5, "kd_gain_top5": k.torch_top5 - b.torch_top5})
    teacher = df[df.variant == "teacher"]
    if len(teacher):
        for c in comp:
            c["teacher_top1"] = float(teacher.iloc[0].torch_top1)
            c["compression_ratio"] = float(teacher.iloc[0].params_M) / c["params_M"]
    write_csv(comp, RESULTS_DIR / "kd_comparison.csv")
    print(pd.DataFrame(comp).to_string(index=False) if comp else "[bench] no KD comparison available")
    print(f"[bench] wrote {RESULTS_DIR / 'benchmark_results.csv'} and {RESULTS_DIR / 'kd_comparison.csv'}")

    from ..report.plots import make_all_plots
    make_all_plots()


if __name__ == "__main__":
    main()
