"""Shared helpers: seeding, device, logging, checkpoint I/O, metrics, CSV/JSON."""
import csv
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", os.getcwd())).resolve()
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
ONNX_DIR = PROJECT_ROOT / "onnx_models"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()


def ensure_dirs():
    for d in (CKPT_DIR, RESULTS_DIR, ONNX_DIR, LOG_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


class Tee:
    """Duplicate stdout/stderr into a log file (logs are reused as report screenshots)."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return False


def setup_logging(name: str):
    ensure_dirs()
    tee = Tee(LOG_DIR / f"{name}.log")
    sys.stdout = tee
    print(f"\n===== {name} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    return tee


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


@torch.no_grad()
def topk_correct(logits, target, ks=(1, 5)):
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    return [correct[:, : min(k, maxk)].any(dim=1).sum().item() for k in ks]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model):
    """Storage size of the serialized fp32 weights (what is actually written to disk)."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def save_checkpoint(path, model, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "meta": meta}, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(rows, path):
    """Write a list of dicts to CSV (union of keys, stable order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def upsert_csv_row(row: dict, path, key: str):
    """Insert or replace a row (matched on `key`) in a CSV file."""
    path = Path(path)
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get(key) != str(row[key])]
    rows.append(row)
    write_csv(rows, path)


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"
