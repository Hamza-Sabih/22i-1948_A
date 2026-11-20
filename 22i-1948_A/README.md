# Knowledge Distillation & Hardware-Aware NAS for Edge Deployment

**Roll number:** 22i-1948 &nbsp;|&nbsp; **Section:** A

This repository contains two end-to-end pipelines:

* **Q1 – Knowledge distillation (CIFAR-100).** A ResNet-50 teacher is distilled into three
  lightweight students (MobileNetV2, SqueezeNet, EfficientNet-Lite0) with soft-target KD
  (temperature softmax), combined hard/soft-label training and feature-based (FitNets-style)
  distillation. Identically trained no-KD baselines, a unified inference pipeline, ONNX export and
  ONNX Runtime CPU inference are included.
* **Q2 – Hardware-aware NAS (CIFAR-10).** A conditional search space (depth, width, kernel size,
  standard / depthwise-separable / MBConv blocks, activation, input resolution), an automatic
  architecture generator, Bayesian optimisation with Optuna TPE, measured CPU latency / storage
  size / peak inference memory with hard deployment constraints, automatic selection, training of
  the best architecture from scratch, ONNX export and ONNX Runtime inference.

## Repository layout

```
22i-1948_A/
├── README.md, requirements.txt
├── notebooks/                    Colab notebook that runs everything end-to-end
├── src/
│   ├── common/                   data pipeline (on-GPU augmentation), trainer, profiling, utils
│   ├── q1_kd/
│   │   ├── models/teacher.py     ResNet-50 (CIFAR stem)                       [deliverable 1]
│   │   ├── models/students.py    MobileNetV2, SqueezeNet, EfficientNet-Lite0  [deliverable 2]
│   │   ├── kd.py                 KD framework (soft targets, T, hard+soft, feature hints) [4]
│   │   ├── train_teacher.py      teacher training script                     [1]
│   │   ├── train_baseline.py     baseline (no-KD) pipeline                    [3]
│   │   └── train_kd.py           distillation training                       [4]
│   ├── q2_nas/
│   │   ├── search_space.py       searchable space                             [5]
│   │   ├── generator.py          config -> nn.Module generator                [5]
│   │   ├── hardware.py           constraints + resource-aware selection       [6]
│   │   ├── search.py             Optuna TPE hardware-aware search             [6, 7]
│   │   └── train_best.py         train the best architecture from scratch     [8]
│   ├── deploy/
│   │   ├── registry.py           model registry
│   │   ├── inference.py          unified inference (all students + NAS)       [9]
│   │   ├── export_onnx.py        ONNX export + verification                   [10]
│   │   ├── onnx_inference.py     ONNX Runtime CPU inference                   [10]
│   │   └── benchmark.py          automated benchmarking                       [11]
│   └── report/                   plots, log screenshots, PDF report generator [12]
├── checkpoints/
│   ├── q1/teacher/resnet50_teacher.pt
│   ├── q1/baseline/{mobilenetv2,squeezenet,efficientnet_lite0}_baseline.pt
│   ├── q1/kd/{mobilenetv2,squeezenet,efficientnet_lite0}_kd.pt
│   └── q2/nas_best.pt
├── onnx_models/                  *.onnx (6 students + NAS model)
├── results/
│   ├── benchmark_results.csv, kd_comparison.csv, onnx_export.csv, onnx_cpu_inference.csv
│   ├── q1/{history,metrics}/     per-epoch CSVs and final metrics JSON
│   ├── q2/best_architecture.json, search_history.csv, search_history.json, nas_study.db
│   └── plots/                    figures used in the report
├── report/                       22i-1948_A_Technical_Report.pdf
├── screenshots/                  execution screenshots
├── samples/                      demo images for the inference pipeline
└── logs/                         full console logs of every stage
```

## Installation

Python ≥ 3.9 and a CUDA GPU are recommended for training (Google Colab T4 works); inference and
ONNX Runtime run on CPU.

```bash
pip install -r requirements.txt
```

## Step-by-step execution

### Option A – Google Colab (recommended)
1. Upload `notebooks/22i-1948_A.ipynb` to Colab.
2. *Runtime → Change runtime type → T4 GPU*.
3. In the first code cell set `SECTION`, `MODE` (`FULL`, `QUICK` or `SMOKE`) and `USE_DRIVE`.
4. *Runtime → Run all*. With `USE_DRIVE=True` everything is written to Google Drive and every
   training stage resumes automatically after a disconnect (just run all cells again).
5. The last cell creates `22i-1948_A.zip` with the complete repository.

### Option B – command line (from the repository root)
```bash
# Q1 – knowledge distillation (CIFAR-100)
python -m src.q1_kd.train_teacher  --epochs 30 --lr 0.05
python -m src.q1_kd.train_baseline --epochs 30
python -m src.q1_kd.train_kd       --epochs 30 --temperature 4 --alpha 0.9 --beta 1.0

# Q2 – hardware-aware NAS (CIFAR-10)
python -m src.q2_nas.search     --n-trials 30 --trial-epochs 10 \
                                --max-latency-ms 5 --max-size-mb 1.0 --max-peak-mem-mb 6
python -m src.q2_nas.train_best --epochs 60

# Deployment, inference and benchmarking
python -m src.deploy.export_onnx                       # ONNX export + numerical check
python -m src.deploy.onnx_inference                    # ONNX Runtime CPU evaluation
python -m src.deploy.inference --make-samples          # demo images
python -m src.deploy.inference --model all             # unified inference (PyTorch)
python -m src.deploy.inference --model all --backend onnx
python -m src.deploy.benchmark --threads 1             # benchmark CSVs + plots

# Report
python -m src.report.make_report --roll 22i-1948 --section A
```
Add `--smoke` to any training/evaluation command to run it on synthetic data in seconds
(functional test only). Every training script skips work whose checkpoint already exists
(use `--force` to retrain) and resumes from `*_last.pt` after an interruption.

### Using a trained model
```python
from src.deploy.inference import UnifiedClassifier
clf = UnifiedClassifier("efficientnet_lite0_kd", backend="onnx")   # or "torch"
print(clf.predict("samples/cifar100_0_mountain.png", topk=3))
```
Model keys: `{mobilenetv2,squeezenet,efficientnet_lite0}_{baseline,kd}`, `nas_best`
(and `resnet50_teacher` for PyTorch).

## Key implementation details
* **KD loss:** `L = (1-α)·CE(z_s, y) + α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) + β·Σ_h ‖norm(r_h(F_s^h)) − norm(F_t^h)‖²`
  with hint layers at 8×8 (teacher layer3) and 4×4 (teacher layer4); `r_h` = 1×1 conv + BN regressor.
* **Fair comparison:** baselines and distilled students use the same data pipeline, optimiser,
  schedule, epochs and seed; only the loss differs.
* **Hardware gate:** each NAS candidate is profiled on CPU *before* training (median latency at
  batch 1, serialized size, peak memory from `torch.profiler` memory events); violators are
  rejected. Selection: `score = val_top1 − 2·lat/lat_budget − 1·size/size_budget` over feasible
  candidates.
* **GitHub note:** the ResNet-50 checkpoint is ~90 MB; GitHub accepts files < 100 MB but Git LFS
  (`git lfs track "*.pt" "*.onnx"`) is recommended.
