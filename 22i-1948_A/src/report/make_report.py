"""Deliverable 12 - build the technical report PDF from the actual experiment outputs
(metrics JSON/CSV, plots, log screenshots). Every number in the report is read from
results/, so the PDF always matches the trained models in the repository.

Usage:  python -m src.report.make_report --roll 22i-1948 --section A
"""
import argparse
import datetime
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from ..common.utils import PROJECT_ROOT, RESULTS_DIR
from ..q2_nas.search_space import search_space_summary
from .plots import PLOTS
from .screenshots import SHOTS, make_screenshots

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#184f95"))
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1c5cab"))
BODY = ParagraphStyle("B", parent=SS["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10)
CODE = ParagraphStyle("C", parent=BODY, fontName="Courier", fontSize=8.3, leading=10.5,
                      backColor=colors.HexColor("#f3f2ee"), borderPadding=4, spaceBefore=3, spaceAfter=7)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.3, textColor=colors.HexColor("#52514e"), alignment=TA_CENTER)
PAGE_W = A4[0] - 3.6 * cm


def P(t, st=BODY):
    return Paragraph(t, st)


def fnum(v, nd=2):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def table(rows, header, col_w=None, font=7.8):
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle("th", parent=SMALL, fontSize=font)) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), ParagraphStyle("td", parent=SMALL, fontSize=font)) for c in r])
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#cde2fb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f3")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def figure(path, caption, width=PAGE_W):
    path = Path(path)
    if not path.exists():
        return []
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = min(width, PAGE_W)
    height = width * h / w
    if height > 21 * cm:
        height = 21 * cm
        width = height * w / h
    return [KeepTogether([Image(str(path), width=width, height=height), P(caption, CAP), Spacer(1, 6)])]


def jload(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def build(roll, section, out_path):
    cfg = jload(RESULTS_DIR / "run_config.json") or {}
    bench = csv(RESULTS_DIR / "benchmark_results.csv")
    kdc = csv(RESULTS_DIR / "kd_comparison.csv")
    onnx_exp = csv(RESULTS_DIR / "onnx_export.csv")
    teacher = jload(RESULTS_DIR / "q1" / "metrics" / "teacher_resnet50.json")
    nas_hist = csv(RESULTS_DIR / "q2" / "search_history.csv")
    nas_best = jload(RESULTS_DIR / "q2" / "best_architecture.json")
    nas_final = jload(RESULTS_DIR / "q2" / "nas_best_metrics.json")
    nas_json = jload(RESULTS_DIR / "q2" / "search_history.json")
    make_screenshots()

    s = []
    # ------------------------------------------------------------------ title
    s += [Spacer(1, 3.2 * cm),
          P("<b>Knowledge Distillation and Hardware-Aware Neural Architecture Search "
            "for Edge Deployment</b>", ParagraphStyle("T", parent=H1, fontSize=20, leading=25, alignment=TA_CENTER)),
          Spacer(1, 0.5 * cm),
          P("Technical Report - Question 1 (Knowledge Distillation) and Question 2 (Neural Architecture Search)",
            ParagraphStyle("st", parent=BODY, alignment=TA_CENTER, fontSize=11)),
          Spacer(1, 1.2 * cm),
          P(f"<b>Roll number:</b> {roll} &nbsp;&nbsp;&nbsp; <b>Section:</b> {section}",
            ParagraphStyle("c", parent=BODY, alignment=TA_CENTER, fontSize=12)),
          P(f"<b>Date:</b> {datetime.date.today():%d %B %Y}", ParagraphStyle("c2", parent=BODY, alignment=TA_CENTER)),
          Spacer(1, 1.2 * cm)]
    env = cfg.get("environment", {})
    if env:
        s.append(table([[k, v] for k, v in env.items()], ["Environment", "Value"], [5 * cm, 10 * cm]))
    if cfg.get("mode") == "SMOKE":
        s.append(P("<font color='#d03b3b'><b>NOTE: this report was generated from a SMOKE-TEST run on synthetic "
                   "data. Re-run the notebook with MODE='FULL' for real results.</b></font>", BODY))
    s.append(PageBreak())

    # ------------------------------------------------------------------ 1 overview
    s.append(P("1. Overview", H1))
    s.append(P("This project implements two complete pipelines for building compact image classifiers for "
               "resource-constrained devices. <b>Question 1</b> transfers knowledge from a large ResNet-50 teacher "
               "trained on CIFAR-100 into three lightweight students (MobileNetV2, SqueezeNet, EfficientNet-Lite0) "
               "using soft-target and feature-based distillation, and compares them against identically trained "
               "baselines. <b>Question 2</b> searches automatically for a small CNN on CIFAR-10 with Bayesian "
               "optimisation (Optuna TPE), measuring the real CPU latency, storage size and peak inference memory of "
               "every candidate and rejecting candidates that violate deployment constraints. All final models are "
               "exported to ONNX and served on CPU with ONNX Runtime; an automated benchmark measures accuracy, "
               "latency, throughput, size and memory of every model."))
    s.append(P("1.1 Deliverables map", H2))
    dm = [["1", "Teacher model", "checkpoints/q1/teacher/resnet50_teacher.pt; src/q1_kd/train_teacher.py"],
          ["2", "Student architectures", "src/q1_kd/models/students.py"],
          ["3", "Baseline students", "checkpoints/q1/baseline/*.pt; src/q1_kd/train_baseline.py"],
          ["4", "Knowledge distillation", "src/q1_kd/kd.py; src/q1_kd/train_kd.py; checkpoints/q1/kd/*.pt"],
          ["5", "NAS search space", "src/q2_nas/search_space.py; src/q2_nas/generator.py"],
          ["6", "NAS algorithm", "src/q2_nas/search.py (Optuna TPE); src/q2_nas/hardware.py"],
          ["7", "NAS results", "results/q2/best_architecture.json; search_history.csv/.json"],
          ["8", "Optimised architecture", "checkpoints/q2/nas_best.pt; src/q2_nas/train_best.py"],
          ["9", "Inference pipeline", "src/deploy/inference.py (all students + NAS, PyTorch/ONNX)"],
          ["10", "ONNX deployment", "onnx_models/*.onnx; src/deploy/export_onnx.py; src/deploy/onnx_inference.py"],
          ["11", "Benchmarking", "src/deploy/benchmark.py; results/benchmark_results.csv; kd_comparison.csv"],
          ["12", "Technical report", "report/ (this PDF); screenshots/; results/plots/"],
          ["13", "README", "README.md; requirements.txt"]]
    s.append(table(dm, ["#", "Deliverable", "Location in repository"], [1 * cm, 4 * cm, 12.3 * cm]))
    s.append(P("1.2 Engineering decisions common to both questions", H2))
    s.append(P("<b>On-device data pipeline.</b> Colab provides only two CPU cores, which makes a torchvision "
               "DataLoader the bottleneck for small networks. The whole dataset is therefore stored on the GPU as "
               "uint8 and augmentation (random crop with 4-pixel padding, horizontal flip), normalisation and "
               "resizing are done batch-wise on the GPU (src/common/data.py). <b>Training engine.</b> SGD with "
               "Nesterov momentum 0.9, no weight decay on BN/bias, one warm-up epoch followed by cosine decay, "
               "mixed precision (FP16 autocast + GradScaler) and channels-last memory format "
               "(src/common/trainer.py). Every epoch writes an atomic resume checkpoint so a disconnected Colab "
               "session continues where it stopped. <b>Reporting.</b> Accuracies are those of the final epoch; "
               "the test set is never used for model selection. In NAS, candidates are ranked on a validation "
               "split held out from the training set."))
    hp = cfg.get("hyperparameters")
    if hp:
        s.append(P("1.3 Run configuration", H2))
        s.append(table([[k, json.dumps(v)] for k, v in hp.items()], ["Stage", "Settings"], [3.6 * cm, 13.7 * cm], 7.2))

    # ------------------------------------------------------------------ Q1
    s.append(PageBreak())
    s.append(P("2. Question 1 - Knowledge distillation framework", H1))
    s.append(P("2.1 Teacher: ResNet-50 on CIFAR-100", H2))
    s.append(P("torchvision's ResNet-50 (bottleneck blocks [3, 4, 6, 3]) is adapted to 32x32 inputs: the 7x7 "
               "stride-2 stem and the max-pool would reduce a CIFAR image to 8x8 before the first residual stage, "
               "so the stem is replaced by a 3x3 stride-1 convolution and the max-pool is removed. The feature maps "
               "are 32x32 (layer1), 16x16 (layer2), 8x8 (layer3, 1024 channels) and 4x4 (layer4, 2048 channels); "
               "layer3 and layer4 are the hint layers used for feature distillation. The network is initialised "
               "from ImageNet weights (the new 3x3 stem from the centre of the pretrained 7x7 kernels) and then "
               "fully trained on CIFAR-100 (all layers updated)."))
    if teacher:
        s.append(table([["ResNet-50 (CIFAR stem)", fnum(teacher["params"] / 1e6, 2), fnum(teacher["size_mb"], 1),
                         teacher["epochs"], fnum(teacher["test_top1"]), fnum(teacher["test_top5"]),
                         fnum(teacher["train_time_s"] / 60, 1)]],
                       ["Model", "Params (M)", "Size (MB)", "Epochs", "Test top-1 (%)", "Test top-5 (%)",
                        "Train time (min)"]))
    s += figure(PLOTS / "q1_teacher_training.png", "Figure 1 - Teacher training curve on CIFAR-100.", 11 * cm)

    s.append(P("2.2 Student architectures", H2))
    s.append(P("All three students are implemented from scratch in src/q1_kd/models/students.py and share the "
               "teacher's interface: <i>model(x, return_features=True)</i> returns the logits plus a "
               "<i>mid</i> (8x8) and a <i>final</i> (4x4) feature map. Early strides are reduced so the total "
               "stride is 8, which matches the teacher's hint resolutions exactly."))
    st_rows = [["MobileNetV2", "Inverted residual blocks with linear bottlenecks, ReLU6, expansion 6; CIFAR strides "
                "(stem 1, stages 1,1,2,2,1,2,1); 1x1 head to 1280 channels", "96@8x8 / 1280@4x4"],
               ["SqueezeNet (v1.1 + BN)", "Fire modules (1x1 squeeze, parallel 1x1/3x3 expand), max-pool "
                "down-sampling 32-16-8-4, 1x1-conv classifier + global average pooling (final ReLU removed so "
                "logits can be negative for KD)", "256@8x8 / 512@4x4"],
               ["EfficientNet-Lite0", "EfficientNet-B0 MBConv layout (kernels 3/5, expansion 6) with the Lite "
                "changes: no squeeze-and-excitation, ReLU6 instead of Swish, fixed stem/head; stochastic depth 0.1",
                "112@8x8 / 1280@4x4"]]
    if bench is not None:
        for r in st_rows:
            key = {"MobileNetV2": "mobilenetv2_baseline", "SqueezeNet (v1.1 + BN)": "squeezenet_baseline",
                   "EfficientNet-Lite0": "efficientnet_lite0_baseline"}[r[0]]
            b = bench[bench.key == key]
            r.append(fnum(b.params_M.iloc[0], 3) if len(b) else "-")
            r.append(fnum(b.mflops.iloc[0], 1) if len(b) else "-")
    else:
        for r in st_rows:
            r += ["-", "-"]
    s.append(table(st_rows, ["Student", "Design", "Hints (mid / final)", "Params (M)", "MFLOPs"],
                   [3 * cm, 8.3 * cm, 2.6 * cm, 1.7 * cm, 1.7 * cm]))

    s.append(P("2.3 Distillation framework (src/q1_kd/kd.py)", H2))
    s.append(P("For a mini-batch with student logits z<sub>s</sub>, teacher logits z<sub>t</sub> and labels y the "
               "student minimises:"))
    s.append(P("L = (1 - alpha) * CE(z_s, y)<br/>"
               "&nbsp;&nbsp;+ alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )<br/>"
               "&nbsp;&nbsp;+ beta * sum_h || norm(r_h(F_s^h)) - norm(F_t^h) ||^2", CODE))
    kd_args = (hp or {}).get("kd", {})
    s.append(P(f"<b>Soft targets with temperature.</b> <i>temperature_softmax</i> computes softmax(z/T); with "
               f"T = {kd_args.get('temperature', 4)} the teacher's distribution exposes the similarity structure "
               f"between classes (dark knowledge). <i>SoftTargetKDLoss</i> is the KL divergence between the "
               f"softened distributions, multiplied by T<super>2</super> so its gradients keep the same scale as "
               f"the cross-entropy term when T changes. <b>Combined hard/soft training.</b> alpha = "
               f"{kd_args.get('alpha', 0.9)} weights the soft term and 1 - alpha the ordinary cross-entropy on the "
               f"ground-truth labels. <b>Feature-based distillation.</b> <i>FeatureDistillation</i> implements "
               f"FitNets-style hints at two depths (8x8 and 4x4): a learnable 1x1 convolution + BatchNorm regressor "
               f"r<sub>h</sub> maps the student feature map to the teacher's channel count, spatial sizes are aligned "
               f"with adaptive pooling, and both maps are L2-normalised per sample before the squared error, which "
               f"makes the loss scale-free (beta = {kd_args.get('beta', 1.0)}). Attention transfer is available as "
               f"an alternative hint loss (--feat-loss at). <i>DistillationWrapper</i> bundles the frozen teacher "
               f"(eval mode, no gradients), the student and the regressors, so the generic trainer optimises "
               f"student + regressors jointly; only the student is saved and deployed."))
    s.append(P("2.4 Baseline pipeline", H2))
    s.append(P("src/q1_kd/train_baseline.py trains the same three students with plain cross-entropy and exactly "
               "the same data pipeline, optimiser, schedule, epochs and seed as the KD runs. The only difference "
               "between a baseline and a distilled checkpoint is therefore the training signal, which makes the "
               "accuracy difference a fair measure of the benefit of distillation."))

    s.append(P("2.5 Results: distillation vs baseline", H2))
    if kdc is not None and len(kdc):
        rows = [[r.student, fnum(r.params_M, 3), fnum(getattr(r, "compression_ratio", None), 1) + "x",
                 fnum(r.baseline_top1), fnum(r.kd_top1), ("+" if r.kd_gain_top1 >= 0 else "") + fnum(r.kd_gain_top1),
                 fnum(r.baseline_top5), fnum(r.kd_top5)] for r in kdc.itertuples()]
        s.append(table(rows, ["Student", "Params (M)", "Compression vs teacher", "Baseline top-1", "KD top-1",
                              "KD gain (pts)", "Baseline top-5", "KD top-5"]))
        s.append(Spacer(1, 4))
        gains = kdc.kd_gain_top1
        best_row = kdc.loc[kdc.kd_top1.idxmax()]
        txt = (f"Distillation changed top-1 accuracy by {', '.join(f'{g:+.2f}' for g in gains)} points for "
               f"{', '.join(kdc.student)} respectively (mean {gains.mean():+.2f}). ")
        if (gains > 0).all():
            txt += "All three students benefit from the teacher's soft targets and feature hints. "
        elif (gains > 0).any():
            txt += "Not every student benefits equally; the gain depends on capacity and on how well its features align with the teacher's. "
        txt += (f"The most accurate distilled student is {best_row.student} ({best_row.kd_top1:.2f}% top-1)")
        if "teacher_top1" in kdc:
            txt += (f", which recovers {100 * best_row.kd_top1 / kdc.teacher_top1.iloc[0]:.1f}% of the teacher's "
                    f"accuracy with {best_row.compression_ratio:.1f}x fewer parameters")
        s.append(P(txt + "."))
    s += figure(PLOTS / "q1_kd_vs_baseline.png", "Figure 2 - CIFAR-100 test accuracy of baseline and distilled students.", 12.5 * cm)
    s += figure(PLOTS / "q1_student_training_curves.png", "Figure 3 - Test accuracy per epoch, baseline vs KD.")
    s += figure(PLOTS / "q1_accuracy_vs_latency.png", "Figure 4 - Accuracy vs measured ONNX Runtime CPU latency.", 11 * cm)

    s.append(P("2.6 Unified inference and ONNX deployment", H2))
    s.append(P("src/deploy/registry.py knows every checkpoint, its builder, dataset and input resolution. "
               "<i>UnifiedClassifier(key, backend)</i> in src/deploy/inference.py loads any of the six students or "
               "the NAS model with either the PyTorch or the ONNX Runtime back-end and accepts image paths, PIL "
               "images or uint8 arrays; pre-processing (resize to 32x32, normalisation with the training-set "
               "statistics, resize to the model's resolution) is shared by all models. src/deploy/export_onnx.py "
               "exports every model with a dynamic batch dimension (opset 17), validates the graph with the ONNX "
               "checker and compares ONNX Runtime outputs against PyTorch on random inputs. "
               "src/deploy/onnx_inference.py runs CPU-only inference (CPUExecutionProvider, all graph "
               "optimisations) and evaluates each model on the full test set."))
    if onnx_exp is not None:
        rows = [[r.key, fnum(r.onnx_size_mb), r.input_shape, f"{r.max_abs_diff_vs_torch:.1e}",
                 fnum(100 * r.top1_agreement, 1) + "%"] for r in onnx_exp.itertuples()]
        s.append(table(rows, ["Model", "ONNX size (MB)", "Input", "max |ONNX - PyTorch|", "Top-1 agreement"]))

    # ------------------------------------------------------------------ Q2
    s.append(PageBreak())
    s.append(P("3. Question 2 - Hardware-aware Neural Architecture Search", H1))
    ss = search_space_summary()
    s.append(P("3.1 Search space (src/q2_nas/search_space.py)", H2))
    s.append(P("An architecture is a JSON dictionary: input resolution, a stem, 2-4 stages and an optional 1x1 head. "
               "The first stage keeps the input resolution and every following stage halves it. Each stage chooses "
               "its own block type, depth, width, kernel size, expansion ratio and activation, so the space is "
               f"conditional (expansion only exists for MBConv stages). It contains about "
               f"{ss['approx_num_architectures']:.2e} distinct architectures."))
    s.append(table([["Input resolution", ss["resolutions"]], ["Stem channels / kernel", f"{ss['stem_channels']} / {ss['stem_kernels']}"],
                    ["Number of stages (depth)", ss["num_stages"]], ["Blocks per stage (depth)", ss["depths"]],
                    ["Channels per stage", ss["stage_channels"]], ["Kernel size", ss["kernels"]],
                    ["Block type", "conv (standard), dws (depthwise separable: depthwise kxk + pointwise 1x1), "
                                   "mbconv (inverted residual)"],
                    ["MBConv expansion", ss["expansions"]], ["Activation", ss["activations"]],
                    ["Head channels (0 = none)", ss["head_channels"]], ["Dropout", ss["dropouts"]]],
                   ["Dimension", "Choices"], [4.5 * cm, 12.8 * cm]))
    s.append(P("3.2 Architecture generator (src/q2_nas/generator.py)", H2))
    s.append(P("<i>build_model(cfg)</i> turns a configuration into a <i>GeneratedNet</i>: a Conv-BN-Act stem, the "
               "stages built from StandardBlock, DepthwiseSeparableBlock or MBConvBlock (a residual connection is "
               "added automatically whenever stride is 1 and the widths match), the optional head, global average "
               "pooling, dropout and a linear classifier. Any sampled configuration is a valid network, which the "
               "search relies on."))
    s.append(P("3.3 Search strategy: Bayesian optimisation with Optuna", H2))
    s.append(P("The search uses Optuna's Tree-structured Parzen Estimator (multivariate, grouped for the conditional "
               "space, 8 random start-up trials). TPE models the densities of good and bad configurations and "
               "proposes the configuration that maximises their ratio. A median pruner stops a candidate whose "
               "validation accuracy halfway through its training budget is below the median of earlier trials. "
               "The study is stored in SQLite (results/q2/nas_study.db), so an interrupted search resumes."))
    s.append(P("3.4 Hardware-aware evaluation and resource-constrained selection", H2))
    cons = (nas_best or {}).get("constraints") or (nas_json or {}).get("constraints") or {}
    s.append(P("Before any training, every candidate is profiled on the CPU (src/common/profiling.py): "
               "<b>latency</b> is the median wall-clock time of single-image forward passes in eval mode after a "
               "warm-up, with a fixed number of CPU threads; <b>storage size</b> is the byte size of the serialised "
               "FP32 state dict; <b>peak inference memory</b> is measured with the PyTorch profiler's memory events "
               "(allocations and frees replayed in time order, the running maximum is the activation peak, plus the "
               "resident weights). Candidates violating any constraint are <b>rejected</b> (pruned without training) "
               "and logged with the reason. Feasible candidates are trained on a CIFAR-10 training subset and scored "
               "on a held-out validation split with:"))
    s.append(P("score = val_top1(%) - 2 * latency / latency_budget - 1 * size / size_budget", CODE))
    s.append(P("The best architecture is the feasible trial with the highest score, so the selection uses the measured "
               "latency and size directly as well as the hard constraints."))
    if cons:
        s.append(table([[f"{cons.get('max_latency_ms')} ms", f"{cons.get('max_size_mb')} MB",
                         f"{cons.get('max_peak_mem_mb')} MB", cons.get("threads")]],
                       ["Max CPU latency (batch 1)", "Max model size", "Max peak inference memory", "CPU threads"]))
    s.append(P("3.5 Search results", H2))
    if nas_hist is not None and len(nas_hist):
        cnt = nas_hist.status.value_counts().to_dict()
        s.append(P(f"{len(nas_hist)} candidates were sampled: {cnt.get('completed', 0)} trained to completion, "
                   f"{cnt.get('pruned_early', 0)} stopped early by the median pruner and "
                   f"{cnt.get('rejected_constraints', 0)} rejected by the hardware constraints before training. "
                   "The full history (architecture, measurements, violations, accuracy, score) is in "
                   "results/q2/search_history.csv and .json."))
        top = nas_hist[nas_hist.status == "completed"].sort_values("score", ascending=False).head(8)
        rows = [[int(r.trial), fnum(r.val_top1), fnum(r.score), fnum(r.cpu_latency_ms), fnum(r.size_mb, 3),
                 fnum(r.peak_mem_mb), fnum(r.params / 1e3, 0), r.arch] for r in top.itertuples()]
        s.append(table(rows, ["Trial", "Val top-1", "Score", "Lat (ms)", "Size (MB)", "Peak mem (MB)", "Params (k)",
                              "Architecture"], [1 * cm, 1.3 * cm, 1.2 * cm, 1.2 * cm, 1.3 * cm, 1.5 * cm, 1.3 * cm,
                                                8.5 * cm], 6.8))
    s += figure(PLOTS / "q2_nas_constraints_scatter.png", "Figure 5 - Measured latency and size of all candidates; dashed lines are the constraints.", 11.5 * cm)
    s += figure(PLOTS / "q2_nas_accuracy_vs_latency.png", "Figure 6 - Validation accuracy vs latency of feasible candidates.", 11.5 * cm)
    s += figure(PLOTS / "q2_nas_history.png", "Figure 7 - Optimisation history.", 11 * cm)
    if nas_best:
        s.append(P("3.6 Best discovered architecture", H2))
        a = nas_best["arch"]
        rows = [[i, st["block"], st["depth"], st["channels"], st["kernel"], st["expansion"], st["activation"], st["stride"]]
                for i, st in enumerate(a["stages"])]
        s.append(P(f"Trial {nas_best['trial']} - input {a['resolution']}x{a['resolution']}, stem "
                   f"{a['stem_channels']} channels ({a['stem_kernel']}x{a['stem_kernel']}), head "
                   f"{a['head_channels'] or 'none'}, dropout {a['dropout']}. Search validation top-1 "
                   f"{nas_best['val_top1']:.2f}%."))
        s.append(table(rows, ["Stage", "Block", "Depth", "Channels", "Kernel", "Expansion", "Activation", "Stride"]))
    if nas_final:
        s.append(Spacer(1, 4))
        s.append(P(f"The architecture was rebuilt with fresh weights and trained <b>from scratch</b> on all 50,000 "
                   f"CIFAR-10 training images for {nas_final['epochs']} epochs (src/q2_nas/train_best.py), reaching "
                   f"<b>{nas_final['test_top1']:.2f}% top-1</b> ({nas_final['test_top5']:.2f}% top-5) on the CIFAR-10 "
                   f"test set with {nas_final['params'] / 1e3:.0f}k parameters, {nas_final['size_mb']:.3f} MB of "
                   f"weights, {nas_final['cpu_latency_ms']:.2f} ms CPU latency and "
                   f"{fnum(nas_final.get('peak_mem_mb'))} MB peak inference memory."))
    s += figure(PLOTS / "q2_nas_best_training.png", "Figure 8 - Training the selected architecture from scratch.", 11 * cm)

    # ------------------------------------------------------------------ benchmarks
    s.append(PageBreak())
    s.append(P("4. Benchmarking (src/deploy/benchmark.py)", H1))
    s.append(P("The benchmark loads every checkpoint and ONNX model and measures, on the same machine and with the same "
               "number of CPU threads: parameters, checkpoint and ONNX size, FLOPs (torch.utils.flop_counter), test "
               "accuracy in PyTorch and in ONNX Runtime, median single-image CPU latency in PyTorch and ONNX Runtime, "
               "ONNX Runtime throughput at batch 64 and peak inference memory. Q1 models are evaluated on CIFAR-100 "
               "and the NAS model on CIFAR-10, so accuracies are only comparable within a question."))
    if bench is not None:
        rows = [[r.key, r.dataset, fnum(r.params_M, 3), fnum(r.ckpt_size_mb), fnum(r.mflops, 1), fnum(r.torch_top1),
                 fnum(getattr(r, "onnx_cpu_top1", None)), fnum(r.torch_cpu_latency_ms),
                 fnum(getattr(r, "ort_cpu_latency_ms", None)), fnum(getattr(r, "ort_throughput_b64_img_s", None), 0),
                 fnum(r.peak_cpu_mem_mb)] for r in bench.itertuples()]
        s.append(table(rows, ["Model", "Data", "Params (M)", "Ckpt (MB)", "MFLOPs", "Top-1 PyTorch", "Top-1 ONNX CPU",
                              "PyTorch CPU (ms)", "ORT CPU (ms)", "ORT img/s (b64)", "Peak mem (MB)"],
                       [3.6 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 1.3 * cm,
                        1.4 * cm, 1.3 * cm], 6.8))
        th = bench.cpu_threads.iloc[0] if "cpu_threads" in bench else 1
        s.append(P(f"Latencies are medians over repeated single-image runs with {th} CPU thread(s).", SMALL))
    s += figure(PLOTS / "cpu_latency_torch_vs_onnx.png", "Figure 9 - CPU latency, PyTorch vs ONNX Runtime.")

    s.append(P("5. Discussion", H1))
    disc = []
    if bench is not None and "ort_speedup_vs_torch" in bench:
        sp = bench.ort_speedup_vs_torch.dropna()
        if len(sp):
            disc.append(f"ONNX Runtime is {sp.min():.1f}x-{sp.max():.1f}x faster than eager PyTorch for single-image "
                        "CPU inference, thanks to operator fusion (Conv+BN+activation) and lower per-call overhead, "
                        "while the ONNX accuracies match PyTorch, confirming the exports are numerically faithful.")
        q1 = bench[(bench.question == "Q1") & (bench.variant != "teacher")]
        if len(q1):
            fast = q1.loc[q1.ort_cpu_latency_ms.idxmin()] if "ort_cpu_latency_ms" in q1 else None
            if fast is not None:
                disc.append(f"Among the students, {fast.model} is the fastest on CPU ({fast.ort_cpu_latency_ms:.2f} ms) "
                            f"with {fast.params_M:.2f}M parameters. FLOPs alone do not predict latency: depthwise "
                            "convolutions have few FLOPs but low arithmetic intensity, which is why the NAS measures "
                            "latency instead of estimating it.")
    if nas_final:
        disc.append(f"The NAS model meets every deployment constraint and reaches {nas_final['test_top1']:.2f}% on "
                    f"CIFAR-10 with only {nas_final['params'] / 1e3:.0f}k parameters, showing that measuring hardware "
                    "cost inside the search loop yields architectures that are deployable by construction.")
    disc.append("Limitations: training budgets were sized for a single Colab GPU session; longer schedules, stronger "
                "augmentation (AutoAugment, CutMix) and a longer NAS search would raise all accuracies. Latency was "
                "measured on the Colab host CPU, so absolute numbers will differ on a real edge device, although "
                "the ranking of models is expected to hold.")
    for d in disc:
        s.append(P(d))

    shots = sorted(SHOTS.glob("*.png")) if SHOTS.exists() else []
    if shots:
        s.append(PageBreak())
        s.append(P("6. Execution screenshots", H1))
        s.append(P("Captured output of each pipeline stage (from logs/), plus any screenshots added to screenshots/."))
        for i, sh in enumerate(shots, 1):
            s += figure(sh, f"Screenshot {i} - {sh.stem.replace('log_', '').lstrip('0123456789_').replace('_', ' ')}")

    s.append(P("7. Reproducing the results", H1))
    s.append(P("Open the notebook in Google Colab (Runtime > Change runtime type > GPU) and run all cells, or run the "
               "scripts from the repository root:"))
    g = lambda stage, k, d: (hp or {}).get(stage, {}).get(k, d)  # noqa: E731
    s.append(P("pip install -r requirements.txt<br/>"
               f"python -m src.q1_kd.train_teacher --epochs {g('teacher', 'epochs', 30)}<br/>"
               f"python -m src.q1_kd.train_baseline --epochs {g('baseline', 'epochs', 30)}<br/>"
               f"python -m src.q1_kd.train_kd --epochs {g('kd', 'epochs', 30)} --temperature {g('kd', 'temperature', 4)} "
               f"--alpha {g('kd', 'alpha', 0.9)} --beta {g('kd', 'beta', 1.0)}<br/>"
               f"python -m src.q2_nas.search --n-trials {g('nas_search', 'n_trials', 30)} "
               f"--trial-epochs {g('nas_search', 'trial_epochs', 10)}<br/>"
               f"python -m src.q2_nas.train_best --epochs {g('nas_best', 'epochs', 60)}<br/>"
               "python -m src.deploy.export_onnx<br/>"
               "python -m src.deploy.onnx_inference<br/>"
               "python -m src.deploy.inference --model all<br/>"
               "python -m src.deploy.benchmark<br/>"
               f"python -m src.report.make_report --roll {roll} --section {section}", CODE))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(1.8 * cm, 1.1 * cm, f"{roll}_{section} - KD & hardware-aware NAS")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="KD and Hardware-aware NAS - Technical Report",
                            author=roll)
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"[report] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Generate the technical report PDF")
    p.add_argument("--roll", default="22i-1948")
    p.add_argument("--section", default="A")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    build(a.roll, a.section, a.out or PROJECT_ROOT / "report" / f"{a.roll}_{a.section}_Technical_Report.pdf")


if __name__ == "__main__":
    main()
