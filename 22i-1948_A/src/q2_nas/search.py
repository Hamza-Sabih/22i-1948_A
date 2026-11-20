"""Deliverables 6 & 7 - hardware-aware NAS with Bayesian optimisation (Optuna TPE).

Pipeline for every candidate (one Optuna trial):
  1. TPE sampler proposes an architecture from the search space (search_space.py)
  2. generator.py builds the nn.Module
  3. HARDWARE GATE - measured on CPU before any training:
        median CPU latency (batch 1), serialized model size, peak inference memory
     -> candidates violating a deployment constraint are REJECTED (trial pruned, no training)
  4. feasible candidates are trained on a CIFAR-10 training subset and validated on a
     held-out validation split (the test set is never used during search); a median
     pruner stops clearly bad candidates early
  5. objective = resource-aware score (hardware.constrained_score)
The study is stored in SQLite, so an interrupted Colab session resumes where it stopped.
Outputs: results/q2/search_history.csv/.json, results/q2/best_architecture.json

Usage:  python -m src.q2_nas.search --n-trials 30 --trial-epochs 10
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import warnings

import optuna
import torch

from ..common.data import DeviceLoader, load_cifar, train_val_split
from ..common.trainer import ce_criterion, fit
from ..common.utils import (RESULTS_DIR, fmt_time, get_device, save_json, seed_everything,
                            setup_logging, write_csv)
from .generator import build_model
from .hardware import DeploymentConstraints, check_constraints, constrained_score, hardware_profile
from .search_space import arch_to_string, sample_optuna, search_space_summary

Q2_RES = RESULTS_DIR / "q2"
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def parse():
    p = argparse.ArgumentParser(description="Hardware-aware NAS on CIFAR-10 (Optuna TPE)")
    p.add_argument("--n-trials", type=int, default=30, help="number of TRAINED (feasible) candidates")
    p.add_argument("--max-total-trials", type=int, default=150, help="cap incl. rejected candidates")
    p.add_argument("--trial-epochs", type=int, default=10)
    p.add_argument("--train-subset", type=int, default=20000, help="training images per candidate")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--max-latency-ms", type=float, default=5.0)
    p.add_argument("--max-size-mb", type=float, default=1.0)
    p.add_argument("--max-peak-mem-mb", type=float, default=6.0)
    p.add_argument("--threads", type=int, default=1, help="CPU threads for latency measurement")
    p.add_argument("--latency-runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", default="hw_aware_nas_cifar10")
    p.add_argument("--fresh", action="store_true", help="delete previous study and start over")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def export_history(study, constraints, args):
    rows, full = [], []
    for t in study.trials:
        ua = t.user_attrs
        if "arch" not in ua:
            continue
        hw = ua.get("hw", {})
        row = {"trial": t.number, "status": ua.get("status", t.state.name.lower()),
               "score": t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
               "val_top1": ua.get("val_top1"), "epochs_trained": ua.get("epochs_trained", 0),
               "cpu_latency_ms": hw.get("cpu_latency_ms"), "size_mb": hw.get("size_mb"),
               "peak_mem_mb": hw.get("peak_mem_mb"), "params": hw.get("params"),
               "mflops": (hw.get("flops") or 0) / 1e6, "resolution": ua["arch"]["resolution"],
               "num_stages": len(ua["arch"]["stages"]),
               "total_blocks": sum(s["depth"] for s in ua["arch"]["stages"]),
               "violations": "; ".join(ua.get("violations", [])), "duration_s": ua.get("duration_s"),
               "arch": arch_to_string(ua["arch"]), "arch_json": json.dumps(ua["arch"])}
        rows.append(row)
        full.append({**row, "arch": ua["arch"], "hw": hw})
    write_csv(rows, Q2_RES / "search_history.csv")
    save_json({"constraints": constraints.to_dict(), "args": vars(args), "search_space": search_space_summary(),
               "trials": full}, Q2_RES / "search_history.json")
    return rows


def main():
    args = parse()
    setup_logging("q2_nas_search")
    print(vars(args))
    seed_everything(args.seed)
    device = get_device()
    constraints = DeploymentConstraints(args.max_latency_ms, args.max_size_mb, args.max_peak_mem_mb, args.threads)
    print(f"[nas] constraints: {constraints.to_dict()}")
    print(f"[nas] search space size ~ {search_space_summary()['approx_num_architectures']:.3e} architectures")

    data = load_cifar("cifar10", smoke=args.smoke)
    xtr, ytr, xval, yval = train_val_split(data["x_train"], data["y_train"],
                                           min(args.val_size, len(data["y_train"]) // 5), seed=args.seed)
    n_sub = min(args.train_subset, len(ytr))
    xtr, ytr = xtr[:n_sub], ytr[:n_sub]
    print(f"[nas] candidate training images={len(ytr)}  validation images={len(yval)} (held out from train)")

    Q2_RES.mkdir(parents=True, exist_ok=True)
    # SQLite on a network/FUSE file system (Google Drive) can fail with locking errors,
    # so the study runs on local disk and is mirrored into results/ after every trial.
    db = Q2_RES / "nas_study.db"
    work_db = Path(os.environ.get("NAS_DB_WORKDIR", tempfile.gettempdir())) / f"{args.study_name}.db"
    if args.fresh:
        db.unlink(missing_ok=True)
        work_db.unlink(missing_ok=True)
    if db.exists() and (not work_db.exists() or db.stat().st_mtime > work_db.stat().st_mtime):
        shutil.copy2(db, work_db)  # resume a study saved by an earlier session
    storage = f"sqlite:///{work_db}"
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True,
                                         n_startup_trials=8, constant_liar=False)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, args.trial_epochs // 2))
    study = optuna.create_study(study_name=args.study_name, storage=storage, direction="maximize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    study.set_user_attr("constraints", constraints.to_dict())

    def objective(trial):
        t0 = time.time()
        cfg = sample_optuna(trial)
        trial.set_user_attr("arch", cfg)
        model = build_model(cfg, data["num_classes"])
        hw = hardware_profile(model, cfg["resolution"], constraints, runs=args.latency_runs)
        trial.set_user_attr("hw", hw)
        violations = check_constraints(hw, constraints)
        trial.set_user_attr("violations", violations)
        tag = (f"trial {trial.number:3d} r{cfg['resolution']} params={hw['params'] / 1e3:.0f}k "
               f"lat={hw['cpu_latency_ms']:.2f}ms size={hw['size_mb']:.2f}MB mem={hw['peak_mem_mb'] or 0:.2f}MB")
        if violations:
            trial.set_user_attr("status", "rejected_constraints")
            trial.set_user_attr("duration_s", time.time() - t0)
            print(f"[nas] {tag} -> REJECTED ({'; '.join(violations)})", flush=True)
            raise optuna.TrialPruned()
        model = model.to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        train = DeviceLoader(xtr, ytr, args.batch_size, device, "cifar10", True, cfg["resolution"], seed=trial.number)
        val = DeviceLoader(xval, yval, 1000, device, "cifar10", False, cfg["resolution"])
        state = {"epochs": 0}

        def report(epoch, top1):
            state["epochs"] = epoch + 1
            trial.report(top1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            history, best, _ = fit(model, train, val, device, epochs=args.trial_epochs, lr=args.lr, wd=args.wd,
                                   criterion=ce_criterion(0.0), warmup_epochs=1, run_name=f"trial{trial.number}",
                                   report_fn=report, log_every_epoch=False)
        except optuna.TrialPruned:
            trial.set_user_attr("status", "pruned_early")
            trial.set_user_attr("epochs_trained", state["epochs"])
            trial.set_user_attr("duration_s", time.time() - t0)
            print(f"[nas] {tag} -> pruned by median rule after {state['epochs']} epochs", flush=True)
            raise
        val_top1 = history[-1]["val_top1"]
        score = constrained_score(val_top1, hw, constraints)
        trial.set_user_attr("status", "completed")
        trial.set_user_attr("val_top1", val_top1)
        trial.set_user_attr("epochs_trained", args.trial_epochs)
        trial.set_user_attr("duration_s", time.time() - t0)
        print(f"[nas] {tag} -> val@1={val_top1:.2f} score={score:.2f} ({time.time() - t0:.0f}s)", flush=True)
        return score

    t_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def n_trained():
        return sum(1 for t in study.trials if t.user_attrs.get("status") in ("completed", "pruned_early"))

    while n_trained() < args.n_trials and len(study.trials) < args.max_total_trials:
        study.optimize(objective, n_trials=1, catch=(RuntimeError,))
        export_history(study, constraints, args)
        shutil.copy2(work_db, db)
    rows = export_history(study, constraints, args)
    shutil.copy2(work_db, db)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError("no feasible architecture found - relax the deployment constraints")
    best = max(done, key=lambda t: t.value)  # resource-constrained selection (feasible only)
    result = {"trial": best.number, "score": best.value, "val_top1": best.user_attrs["val_top1"],
              "arch": best.user_attrs["arch"], "arch_string": arch_to_string(best.user_attrs["arch"]),
              "hw": best.user_attrs["hw"], "constraints": constraints.to_dict(),
              "selection_rule": "argmax over feasible trials of val_top1 - 2*lat/lat_budget - 1*size/size_budget",
              "search_time_s": time.time() - t_start}
    save_json(result, Q2_RES / "best_architecture.json")
    stats = {s: sum(1 for r in rows if r["status"] == s) for s in ("completed", "pruned_early", "rejected_constraints")}
    print(f"[nas] trials: {len(rows)} total, {stats}")
    print(f"[nas] BEST trial {best.number}: val@1={result['val_top1']:.2f} score={best.value:.2f} "
          f"lat={result['hw']['cpu_latency_ms']:.2f}ms size={result['hw']['size_mb']:.3f}MB")
    print(f"[nas] arch: {result['arch_string']}")
    print(f"[nas] search time this session {fmt_time(time.time() - t_start)} -> {Q2_RES / 'best_architecture.json'}")


if __name__ == "__main__":
    main()
