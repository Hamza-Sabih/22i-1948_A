"""Deliverable 9 - unified inference pipeline for ALL trained student models
(baseline + distilled MobileNetV2 / SqueezeNet / EfficientNet-Lite0) and the NAS model.

One class, two back-ends:
    clf = UnifiedClassifier("mobilenetv2_kd", backend="torch")   # PyTorch (CPU or GPU)
    clf = UnifiedClassifier("nas_best", backend="onnx")          # ONNX Runtime (CPU)
    clf.predict("samples/cifar100_0_cat.png", topk=3)

Accepted inputs: image path(s), PIL image(s), uint8 numpy HWC / NHWC arrays.
Pre-processing is identical for every model (RGB -> 32x32 -> normalise with the
training-set statistics -> resize to the model's input resolution).

CLI:
    python -m src.deploy.inference --list
    python -m src.deploy.inference --make-samples
    python -m src.deploy.inference --model all --images samples/*.png
    python -m src.deploy.inference --model mobilenetv2_kd --backend onnx --images samples/cifar100_*.png
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from ..common.data import load_cifar, preprocess_numpy
from ..common.utils import PROJECT_ROOT, setup_logging
from .registry import available_keys, describe, load_model, onnx_path


def to_uint8_batch(images):
    """Normalise any supported input to a uint8 NHWC 32x32 RGB batch."""
    from PIL import Image
    if isinstance(images, (str, Path, Image.Image)):
        images = [images]
    if isinstance(images, np.ndarray):
        arr = images if images.ndim == 4 else images[None]
        if arr.shape[1:3] != (32, 32):
            arr = np.stack([np.array(Image.fromarray(a).resize((32, 32), Image.BILINEAR)) for a in arr])
        return arr.astype(np.uint8)
    out = []
    for im in images:
        if isinstance(im, (str, Path)):
            im = Image.open(im)
        out.append(np.array(im.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.uint8))
    return np.stack(out)


class UnifiedClassifier:
    def __init__(self, key, backend="torch", device=None, threads=1):
        self.key, self.backend = key, backend
        model, meta = load_model(key)
        self.meta, self.classes = meta, meta["classes"]
        self.dataset, self.resolution = meta["dataset"], meta["resolution"]
        if backend == "onnx":
            from .onnx_inference import OnnxClassifier
            if not onnx_path(key).exists():
                raise FileNotFoundError(f"{onnx_path(key)} missing - run src.deploy.export_onnx first")
            self.ort = OnnxClassifier(key, threads, self.dataset, self.resolution, self.classes)
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict_proba(self, images):
        batch = to_uint8_batch(images)
        if self.backend == "onnx":
            x = preprocess_numpy(batch, self.dataset, self.resolution).numpy()
            z = self.ort.logits(x)
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            return p / p.sum(1, keepdims=True)
        x = preprocess_numpy(batch, self.dataset, self.resolution).to(self.device)
        return torch.softmax(self.model(x).float(), 1).cpu().numpy()

    def predict(self, images, topk=3):
        p = self.predict_proba(images)
        idx = np.argsort(-p, 1)[:, :topk]
        return [[(self.classes[j], float(p[i, j])) for j in idx[i]] for i in range(len(p))]


def make_samples(n=8, out_dir=None, smoke=False):
    """Save a few CIFAR-10 / CIFAR-100 test images as PNG files for the demo."""
    from PIL import Image
    out_dir = Path(out_dir or PROJECT_ROOT / "samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in ("cifar100", "cifar10"):
        d = load_cifar(ds, smoke=smoke)
        for i in range(n):
            label = d["classes"][d["y_test"][i]]
            Image.fromarray(d["x_test"][i]).save(out_dir / f"{ds}_{i}_{label}.png")
    print(f"[samples] wrote {2 * n} images to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Unified inference for all student models and the NAS model")
    p.add_argument("--model", default="all", help="registry key or 'all'")
    p.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    p.add_argument("--images", nargs="*", default=None)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--list", action="store_true")
    p.add_argument("--make-samples", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    setup_logging("unified_inference")
    if args.list:
        for k in available_keys():
            print(f"{k:28s} {describe(k)}")
        return
    if args.make_samples:
        make_samples(smoke=args.smoke)
        return
    keys = available_keys() if args.model == "all" else [args.model]
    files = []
    for pat in args.images or [str(PROJECT_ROOT / "samples" / "*.png")]:
        files += sorted(glob.glob(pat))
    for key in keys:
        clf = UnifiedClassifier(key, args.backend)
        # each model only classifies images of its own dataset (CIFAR-100 students, CIFAR-10 NAS)
        sel = [f for f in files if Path(f).name.startswith(clf.dataset + "_")] or files
        preds = clf.predict(sel, args.topk)
        hits = 0
        print(f"\n=== {key} ({args.backend}, {clf.dataset}, {clf.resolution}x{clf.resolution}) ===")
        for f, pr in zip(sel, preds):
            truth = Path(f).stem.split("_", 2)[-1] if Path(f).stem.count("_") >= 2 else "?"
            hits += pr[0][0] == truth
            top = ", ".join(f"{c} {s:.2f}" for c, s in pr)
            print(f"  {Path(f).name:36s} -> {top}")
        print(f"  top-1 correct on samples: {hits}/{len(sel)}")


if __name__ == "__main__":
    main()
