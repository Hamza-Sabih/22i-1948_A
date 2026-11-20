"""Figures for the technical report (written to results/plots/)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..common.utils import RESULTS_DIR  # noqa: E402

PLOTS = RESULTS_DIR / "plots"
# fixed categorical order (validated palette): blue, orange, aqua, yellow, magenta, green, violet
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#6250d6")
CRITICAL, MUTED, INK, INK2 = "#d03b3b", "#a3a29c", "#0b0b0b", "#52514e"
STUDENTS = [("mobilenetv2", "MobileNetV2"), ("squeezenet", "SqueezeNet"), ("efficientnet_lite0", "EfficientNet-Lite0")]

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight"})


def _read(path):
    return pd.read_csv(path) if path.exists() else None


def _save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name)
    plt.close(fig)
    print(f"[plots] {PLOTS / name}")


def teacher_curve():
    h = _read(RESULTS_DIR / "q1" / "history" / "teacher_resnet50.csv")
    if h is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h.epoch, h.train_top1, color=BLUE, label="train top-1")
    ax.plot(h.epoch, h.val_top1, color=ORANGE, label="test top-1")
    ax.set(xlabel="epoch", ylabel="accuracy (%)", title="ResNet-50 teacher on CIFAR-100")
    ax.legend()
    _save(fig, "q1_teacher_training.png")


def student_curves():
    hist = RESULTS_DIR / "q1" / "history"
    tm = RESULTS_DIR / "q1" / "metrics" / "teacher_resnet50.json"
    t_acc = json.load(open(tm))["test_top1"] if tm.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    any_ = False
    for ax, (k, name) in zip(axes, STUDENTS):
        for variant, color, label in (("baseline", BLUE, "baseline (CE only)"), ("kd", ORANGE, "distilled (KD)")):
            h = _read(hist / f"{k}_{variant}.csv")
            if h is not None:
                ax.plot(h.epoch, h.val_top1, color=color, label=label)
                any_ = True
        if t_acc is not None:
            ax.axhline(t_acc, color=MUTED, ls="--", lw=1.5, label="teacher")
        ax.set(title=name, xlabel="epoch")
    axes[0].set_ylabel("CIFAR-100 test top-1 (%)")
    axes[0].legend(loc="lower right")
    if any_:
        _save(fig, "q1_student_training_curves.png")
    else:
        plt.close(fig)


def kd_bars():
    c = _read(RESULTS_DIR / "kd_comparison.csv")
    if c is None or not len(c):
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = range(len(c))
    w = 0.38
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], c.baseline_top1, w, color=BLUE, label="baseline")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], c.kd_top1, w, color=ORANGE, label="distilled (KD)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")
    if "teacher_top1" in c:
        ax.axhline(c.teacher_top1.iloc[0], color=MUTED, ls="--", lw=1.5, label="ResNet-50 teacher")
    ax.set_xticks(list(x), c.student)
    ax.set(ylabel="CIFAR-100 test top-1 (%)", title="Knowledge distillation vs baseline training")
    ax.set_ylim(0, max(c.kd_top1.max(), c.baseline_top1.max(), c.get("teacher_top1", pd.Series([0])).max()) * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    _save(fig, "q1_kd_vs_baseline.png")


def latency_bars():
    b = _read(RESULTS_DIR / "benchmark_results.csv")
    if b is None or "ort_cpu_latency_ms" not in b:
        return
    b = b[b.ort_cpu_latency_ms.notna()]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = range(len(b))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], b.torch_cpu_latency_ms, w, color=BLUE, label="PyTorch CPU")
    ax.bar([i + w / 2 + 0.01 for i in x], b.ort_cpu_latency_ms, w, color=AQUA, label="ONNX Runtime CPU")
    ax.set_xticks(list(x), [k.replace("_", "\n", 1) for k in b.key], fontsize=8)
    ax.set(ylabel="median latency, batch 1 (ms)", title="CPU inference latency (single image)")
    ax.legend()
    _save(fig, "cpu_latency_torch_vs_onnx.png")

    q1 = b[(b.question == "Q1") & b.onnx_cpu_top1.notna()]
    if len(q1):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for (variant, color, label) in (("baseline", BLUE, "baseline"), ("kd", ORANGE, "distilled")):
            s = q1[q1.variant == variant]
            ax.scatter(s.ort_cpu_latency_ms, s.onnx_cpu_top1, s=60, color=color, label=label,
                       edgecolor="#fcfcfb", linewidth=2, zorder=3)
            for _, r in s.iterrows():
                ax.annotate(r.model, (r.ort_cpu_latency_ms, r.onnx_cpu_top1), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color=INK2)
        ax.set(xlabel="ONNX Runtime CPU latency (ms)", ylabel="CIFAR-100 top-1 (%)",
               title="Students: accuracy vs CPU latency")
        ax.legend()
        _save(fig, "q1_accuracy_vs_latency.png")


def nas_plots():
    h = _read(RESULTS_DIR / "q2" / "search_history.csv")
    if h is None or not len(h):
        return
    cons = json.load(open(RESULTS_DIR / "q2" / "search_history.json"))["constraints"]
    styles = {"completed": (BLUE, "trained & feasible"), "pruned_early": (YELLOW, "pruned (median rule)"),
              "rejected_constraints": (CRITICAL, "rejected by constraints")}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for st, (color, label) in styles.items():
        s = h[h.status == st]
        ax.scatter(s.cpu_latency_ms, s.size_mb, s=40, color=color, label=f"{label} ({len(s)})",
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    ax.axvline(cons["max_latency_ms"], color=INK2, ls="--", lw=1)
    ax.axhline(cons["max_size_mb"], color=INK2, ls="--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="measured CPU latency (ms, log)", ylabel="model size (MB, log)",
           title="NAS candidates vs deployment constraints")
    ax.legend(fontsize=8)
    _save(fig, "q2_nas_constraints_scatter.png")

    done = h[h.status == "completed"]
    if len(done):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        sc = ax.scatter(done.cpu_latency_ms, done.val_top1, c=done.size_mb, cmap="Blues", vmin=0,
                        s=55, edgecolor=INK2, linewidth=0.6, zorder=3)
        best = done.loc[done.score.idxmax()]
        ax.scatter([best.cpu_latency_ms], [best.val_top1], s=180, facecolor="none", edgecolor=ORANGE, linewidth=2.5,
                   label=f"selected (trial {int(best.trial)})", zorder=4)
        fig.colorbar(sc, ax=ax, label="size (MB)")
        ax.set(xlabel="measured CPU latency (ms)", ylabel="validation top-1 (%)", title="Feasible candidates")
        ax.legend(loc="lower right")
        _save(fig, "q2_nas_accuracy_vs_latency.png")

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.scatter(done.trial, done.score, color=BLUE, s=30, label="trial score", zorder=3)
        d2 = done.sort_values("trial")
        ax.step(d2.trial, d2.score.cummax(), where="post", color=ORANGE, label="best so far")
        ax.set(xlabel="trial", ylabel="resource-aware score", title="Optuna (TPE) optimisation history")
        ax.legend()
        _save(fig, "q2_nas_history.png")

    hb = _read(RESULTS_DIR / "q2" / "history_nas_best.csv")
    if hb is not None:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(hb.epoch, hb.train_top1, color=BLUE, label="train top-1")
        ax.plot(hb.epoch, hb.val_top1, color=ORANGE, label="test top-1")
        ax.set(xlabel="epoch", ylabel="accuracy (%)", title="Best NAS architecture trained from scratch (CIFAR-10)")
        ax.legend()
        _save(fig, "q2_nas_best_training.png")


def make_all_plots():
    for fn in (teacher_curve, student_curves, kd_bars, latency_bars, nas_plots):
        try:
            fn()
        except Exception as ex:  # a missing result must not stop the pipeline
            print(f"[plots] {fn.__name__} skipped: {ex}")


if __name__ == "__main__":
    make_all_plots()
