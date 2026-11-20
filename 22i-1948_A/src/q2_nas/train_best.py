"""Deliverable 8 - rebuild the best discovered architecture and train it FROM SCRATCH
on the full CIFAR-10 training set; evaluate on the CIFAR-10 test set.

Usage:  python -m src.q2_nas.train_best --epochs 60
"""
import argparse
import time

import torch

from ..common.data import DeviceLoader, load_cifar
from ..common.trainer import ce_criterion, evaluate, fit
from ..common.utils import (CKPT_DIR, RESULTS_DIR, fmt_time, get_device, load_json, save_checkpoint,
                            save_json, seed_everything, setup_logging)
from .generator import build_model
from .hardware import DeploymentConstraints, check_constraints, hardware_profile
from .search_space import arch_to_string

Q2_CKPT = CKPT_DIR / "q2"
Q2_RES = RESULTS_DIR / "q2"


def main():
    p = argparse.ArgumentParser(description="Train the best NAS architecture from scratch on CIFAR-10")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    setup_logging("q2_train_best")
    print(vars(args))
    out = Q2_CKPT / "nas_best.pt"
    if out.exists() and not args.force:
        print(f"[nas-best] checkpoint exists at {out} - skipping (use --force to retrain)")
        return
    best = load_json(Q2_RES / "best_architecture.json")
    cfg = best["arch"]
    print(f"[nas-best] trial {best['trial']} | {arch_to_string(cfg)}")
    seed_everything(args.seed)
    device = get_device()
    data = load_cifar("cifar10", smoke=args.smoke)
    model = build_model(cfg, data["num_classes"])  # fresh random initialisation (from scratch)
    c = DeploymentConstraints(**best["constraints"])
    hw = hardware_profile(model, cfg["resolution"], c, runs=100, warmup=10)
    print(f"[nas-best] hardware: {hw} | violations: {check_constraints(hw, c) or 'none'}")
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    res = cfg["resolution"]
    train = DeviceLoader(data["x_train"], data["y_train"], args.batch_size, device, "cifar10", True, res, seed=args.seed)
    test = DeviceLoader(data["x_test"], data["y_test"], 1000, device, "cifar10", False, res)
    t0 = time.time()
    history, best_ep, _ = fit(model, train, test, device, epochs=args.epochs, lr=args.lr, wd=args.wd,
                              criterion=ce_criterion(0.0), warmup_epochs=2, run_name="nas_best",
                              resume_path=Q2_CKPT / "nas_best_last.pt", history_csv=Q2_RES / "history_nas_best.csv")
    final = evaluate(model, test, device)
    meta = {"arch": "nas", "arch_config": cfg, "dataset": "cifar10", "num_classes": data["num_classes"],
            "classes": data["classes"], "resolution": res, "test_top1": final["top1"], "test_top5": final["top5"],
            "search_val_top1": best["val_top1"], "hw": hw, "epochs": args.epochs}
    save_checkpoint(out, model, meta)
    save_json({"model": "nas_best", "trial": best["trial"], "arch_string": arch_to_string(cfg),
               "test_top1": final["top1"], "test_top5": final["top5"], "best_epoch_top1": best_ep["top1"],
               "search_val_top1": best["val_top1"], "epochs": args.epochs, "train_time_s": time.time() - t0, **hw},
              Q2_RES / "nas_best_metrics.json")
    print(f"[nas-best] final test top1={final['top1']:.2f} top5={final['top5']:.2f} "
          f"(time {fmt_time(time.time() - t0)}) -> {out}")
    (Q2_CKPT / "nas_best_last.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
