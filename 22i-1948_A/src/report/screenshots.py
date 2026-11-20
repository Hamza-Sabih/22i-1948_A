"""Render terminal-style screenshots (PNG) of the key execution logs, so the report
contains captured evidence of every pipeline stage. Real Colab screenshots can be
added to the `screenshots/` folder as well; the report includes every PNG there."""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common.utils import LOG_DIR, PROJECT_ROOT  # noqa: E402

SHOTS = PROJECT_ROOT / "screenshots"

# (log file, title, regex of lines to keep, max lines)
SPECS = [
    ("q1_train_teacher", "Q1 - ResNet-50 teacher training", r"\[teacher\]", 26),
    ("q1_train_baseline", "Q1 - baseline student training (no KD)", r"final test|params=", 12),
    ("q1_train_kd", "Q1 - knowledge-distillation training", r"\[kd\]|final test|params=| ep +\d+/", 30),
    ("export_onnx", "ONNX export + numerical verification", r"\[onnx\]", 12),
    ("onnx_cpu_inference", "ONNX Runtime CPU inference", r"\[ort-cpu\]", 12),
    ("unified_inference", "Unified inference pipeline", r"===|->|correct", 30),
    ("q2_nas_search", "Q2 - hardware-aware NAS search (Optuna TPE)", r"\[nas\]", 34),
    ("q2_train_best", "Q2 - best architecture trained from scratch", r"\[nas-best\]| ep +\d+/", 30),
    ("benchmark", "Automated benchmarking", r"\[bench\]", 14),
]


def _last_useful_run(text, pattern):
    """Most recent run section that actually did work (re-runs that only print
    'checkpoint exists - skipping' are ignored)."""
    parts = re.split(r"\n(?====== )", text)
    for part in reversed(parts):
        hits = [ln for ln in part.splitlines() if re.search(pattern, ln) and "skipping" not in ln]
        if len(hits) >= 2:
            return part
    return parts[-1] if parts else text


def render(lines, title, path, width=118):
    lines = [ln if len(ln) <= width else ln[: width - 3] + "..." for ln in lines]
    h = 0.6 + 0.165 * (len(lines) + 2)
    fig = plt.figure(figsize=(12, h))
    fig.patch.set_facecolor("#1a1a19")
    fig.text(0.012, 1 - 0.32 / h, f"$ {title}", color="#86b6ef", family="monospace", fontsize=9.5, va="top",
             weight="bold")
    fig.text(0.012, 1 - 0.62 / h, "\n".join(lines), color="#f0efec", family="monospace", fontsize=7.6, va="top",
             linespacing=1.32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_screenshots():
    made = []
    for i, (name, title, pattern, n) in enumerate(SPECS, 1):
        f = LOG_DIR / f"{name}.log"
        if not f.exists():
            continue
        text = _last_useful_run(f.read_text(errors="ignore"), pattern)
        keep = [ln.rstrip() for ln in text.splitlines() if re.search(pattern, ln)]
        if len(keep) > n:  # keep the head and the tail of long logs
            keep = keep[: n // 3] + ["   ..."] + keep[-(n - n // 3 - 1):]
        if not keep:
            continue
        out = SHOTS / f"log_{i:02d}_{name}.png"
        render(keep, title, out)
        made.append(out)
        print(f"[screenshots] {out}")
    return made


if __name__ == "__main__":
    make_screenshots()
