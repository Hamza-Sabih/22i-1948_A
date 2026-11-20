"""Model registry: one place that knows every trained model, its checkpoint,
builder, dataset and input resolution (used by inference, ONNX export, benchmarks)."""
from pathlib import Path

from ..common.utils import CKPT_DIR, ONNX_DIR, load_checkpoint
from ..q1_kd.models.students import PRETTY, STUDENTS, build_student
from ..q1_kd.models.teacher import build_teacher
from ..q2_nas.generator import build_model as build_nas


def all_model_keys(include_teacher=False):
    keys = ["resnet50_teacher"] if include_teacher else []
    keys += [f"{s}_{v}" for v in ("baseline", "kd") for s in STUDENTS]
    keys.append("nas_best")
    return keys


def ckpt_path(key) -> Path:
    if key == "resnet50_teacher":
        return CKPT_DIR / "q1" / "teacher" / "resnet50_teacher.pt"
    if key == "nas_best":
        return CKPT_DIR / "q2" / "nas_best.pt"
    name, variant = key.rsplit("_", 1)
    return CKPT_DIR / "q1" / variant / f"{name}_{variant}.pt"


def onnx_path(key) -> Path:
    return ONNX_DIR / f"{key}.onnx"


def available_keys(include_teacher=False):
    return [k for k in all_model_keys(include_teacher) if ckpt_path(k).exists()]


def describe(key):
    if key == "resnet50_teacher":
        return {"model": "ResNet-50", "variant": "teacher", "question": "Q1", "dataset": "cifar100"}
    if key == "nas_best":
        return {"model": "NAS-best", "variant": "nas", "question": "Q2", "dataset": "cifar10"}
    name, variant = key.rsplit("_", 1)
    return {"model": PRETTY[name], "variant": variant, "question": "Q1", "dataset": "cifar100"}


def load_model(key, map_location="cpu"):
    """Return (eval-mode model on CPU, meta dict) for a registry key."""
    path = ckpt_path(key)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for '{key}' at {path}")
    ck = load_checkpoint(path, map_location)
    meta = ck["meta"]
    if key == "resnet50_teacher":
        model = build_teacher(meta["num_classes"], pretrained=False)
    elif key == "nas_best":
        model = build_nas(meta["arch_config"], meta["num_classes"])
    else:
        model = build_student(meta["arch"], meta["num_classes"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), meta
