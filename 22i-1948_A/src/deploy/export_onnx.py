"""Deliverable 10 - export trained student models and the NAS model to ONNX and
verify each export numerically against PyTorch with ONNX Runtime (CPU).

Usage:  python -m src.deploy.export_onnx                # all available models
        python -m src.deploy.export_onnx --models mobilenetv2_kd nas_best
"""
import argparse
import warnings

import numpy as np
import onnx
import torch

from ..common.profiling import ort_session
from ..common.utils import ONNX_DIR, RESULTS_DIR, setup_logging, write_csv
from .registry import available_keys, describe, load_model, onnx_path

OPSET = 17


def export_one(model, resolution, path, opset=OPSET):
    """Export with a dynamic batch dimension. Uses the TorchScript exporter
    (stable for CNNs, single self-contained file); falls back to the torch.export
    based exporter if the installed PyTorch no longer provides it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, resolution, resolution)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (x,), str(path), input_names=["input"], output_names=["logits"],
                              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                              opset_version=opset, do_constant_folding=True, dynamo=False)
            exporter = "torchscript"
        except Exception as ex:
            print(f"  TorchScript exporter failed ({ex}); using dynamo exporter")
            batch = torch.export.Dim("batch", min=1, max=4096)
            prog = torch.onnx.export(model, (x,), dynamo=True, input_names=["input"], output_names=["logits"],
                                     dynamic_shapes=({0: batch},), opset_version=max(opset, 18))
            prog.save(str(path), external_data=False)
            exporter = "dynamo"
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    return exporter


def verify(model, path, resolution, n=16):
    x = torch.randn(n, 3, resolution, resolution)
    with torch.inference_mode():
        ref = model(x).numpy()
    sess = ort_session(path, threads=2)
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    rel = diff / max(1.0, float(np.abs(ref).max()))
    return diff, rel, float((out.argmax(1) == ref.argmax(1)).mean())


def main():
    p = argparse.ArgumentParser(description="Export models to ONNX and verify with ONNX Runtime")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--include-teacher", action="store_true")
    p.add_argument("--opset", type=int, default=OPSET)
    args = p.parse_args()
    setup_logging("export_onnx")
    keys = args.models or available_keys(args.include_teacher)
    rows = []
    for key in keys:
        model, meta = load_model(key)
        res = meta["resolution"]
        path = onnx_path(key)
        exporter = export_one(model, res, path, args.opset)
        diff, rel, agree = verify(model, path, res)
        size = path.stat().st_size / 2 ** 20
        rows.append({"key": key, **describe(key), "onnx_file": str(path.relative_to(ONNX_DIR.parent)),
                     "onnx_size_mb": size, "opset": args.opset, "exporter": exporter, "input_shape": f"Nx3x{res}x{res}",
                     "max_abs_diff_vs_torch": diff, "max_rel_diff_vs_torch": rel,
                     "top1_agreement": agree})
        status = "OK" if rel < 1e-3 and agree == 1.0 else "CHECK"
        print(f"[onnx] {key:28s} -> {path.name:32s} {size:7.2f}MB  max|diff|={diff:.2e}  agree={agree:.2f}  {status}")
    write_csv(rows, RESULTS_DIR / "onnx_export.csv")
    print(f"[onnx] wrote {RESULTS_DIR / 'onnx_export.csv'}")


if __name__ == "__main__":
    main()
