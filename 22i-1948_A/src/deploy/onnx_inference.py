"""Deliverable 10 - CPU inference with ONNX Runtime.

`OnnxClassifier` wraps an ONNX Runtime CPU session (CPUExecutionProvider, full graph
optimisation, configurable threads). The CLI evaluates every exported model on the
corresponding test set on CPU and measures latency / throughput.

Usage:  python -m src.deploy.onnx_inference                       # all exported models
        python -m src.deploy.onnx_inference --models nas_best --n-test 2000 --threads 2
"""
import argparse
import time

import numpy as np

from ..common.data import load_cifar, preprocess_numpy
from ..common.profiling import ort_cpu_latency, ort_session
from ..common.utils import RESULTS_DIR, load_checkpoint, setup_logging, write_csv
from .registry import all_model_keys, ckpt_path, describe, onnx_path


class OnnxClassifier:
    def __init__(self, key_or_path, threads=1, dataset=None, resolution=None, classes=None):
        path = onnx_path(key_or_path) if not str(key_or_path).endswith(".onnx") else key_or_path
        self.key = str(key_or_path)
        self.session = ort_session(path, threads)
        self.input_name = self.session.get_inputs()[0].name
        meta = {}
        if dataset is None or resolution is None or classes is None:
            ck = ckpt_path(self.key)
            meta = load_checkpoint(ck)["meta"] if ck.exists() else {}
        self.dataset = dataset or meta.get("dataset", "cifar100")
        self.resolution = resolution or meta.get("resolution", 32)
        self.classes = classes or meta.get("classes")

    def logits(self, x_float_nchw: np.ndarray, batch_size=256):
        outs = []
        for i in range(0, len(x_float_nchw), batch_size):
            outs.append(self.session.run(None, {self.input_name: x_float_nchw[i:i + batch_size]})[0])
        return np.concatenate(outs)

    def predict_uint8(self, images_nhwc_uint8, topk=5):
        x = preprocess_numpy(images_nhwc_uint8, self.dataset, self.resolution).numpy()
        z = self.logits(x)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        idx = np.argsort(-p, 1)[:, :topk]
        return idx, np.take_along_axis(p, idx, 1)


def evaluate_onnx(clf: OnnxClassifier, x_uint8, y, batch_size=256):
    t0 = time.perf_counter()
    x = preprocess_numpy(x_uint8, clf.dataset, clf.resolution).numpy()
    z = clf.logits(x, batch_size)
    elapsed = time.perf_counter() - t0
    top5 = np.argsort(-z, 1)[:, :5]
    return {"top1": 100.0 * float((top5[:, 0] == y).mean()), "top5": 100.0 * float((top5 == y[:, None]).any(1).mean()),
            "throughput_img_s": len(y) / elapsed}


def main():
    p = argparse.ArgumentParser(description="ONNX Runtime CPU inference / evaluation")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n-test", type=int, default=10000, help="test images to evaluate (max 10000)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    setup_logging("onnx_cpu_inference")
    keys = args.models or [k for k in all_model_keys(True) if onnx_path(k).exists()]
    data_cache, rows = {}, []
    for key in keys:
        clf = OnnxClassifier(key, threads=args.threads)
        if clf.dataset not in data_cache:
            data_cache[clf.dataset] = load_cifar(clf.dataset, smoke=args.smoke)
        d = data_cache[clf.dataset]
        n = min(args.n_test, len(d["y_test"]))
        ev = evaluate_onnx(clf, d["x_test"][:n], d["y_test"][:n])
        lat = ort_cpu_latency(clf.session, (1, 3, clf.resolution, clf.resolution), runs=100)
        rows.append({"key": key, **describe(key), "n_test": n, "threads": args.threads, "onnx_top1": ev["top1"],
                     "onnx_top5": ev["top5"], "batch_throughput_img_s": ev["throughput_img_s"],
                     "latency_b1_median_ms": lat["median_ms"], "latency_b1_p90_ms": lat["p90_ms"]})
        print(f"[ort-cpu] {key:28s} top1={ev['top1']:6.2f} top5={ev['top5']:6.2f} "
              f"b1-latency={lat['median_ms']:.3f}ms throughput={ev['throughput_img_s']:.0f} img/s")
    write_csv(rows, RESULTS_DIR / "onnx_cpu_inference.csv")
    print(f"[ort-cpu] wrote {RESULTS_DIR / 'onnx_cpu_inference.csv'}")


if __name__ == "__main__":
    main()
